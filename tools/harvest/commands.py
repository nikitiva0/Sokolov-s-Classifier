"""Команды tools/harvest_faces.py: reference, calibrate, video, images, sheet, dedup-v1, set-status, stats."""
from __future__ import annotations
import fcntl
import json
import shlex
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

from . import REFERENCE, REFERENCE_STATS, REPO, THRESHOLDS, Layout, slug
from . import net, sheets
from .geometry import crop_face, timestamps
from .labeling import classify, identity, load_thresholds
from .manifest import (append_rows, natural_key, next_index, read_rows, refresh_sources,
                       update_rows, write_json)

V1 = REPO / 'data/raw/sokolov_positive_v1'
SCRATCH = REPO / 'data/raw/_scratch'


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def quantiles(values):
    import numpy as np
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return {'n': 0}
    q = np.quantile(v, [0, .01, .05, .5, .95, .99, 1])
    return {'n': int(v.size), **{k: round(float(x), 4) for k, x in
                                  zip(['min', 'p1', 'p5', 'median', 'p95', 'p99', 'max'], q)}}


def unit(v):
    import numpy as np
    return (v / np.linalg.norm(v, axis=-1, keepdims=True)).astype(np.float32)


def load_reference():
    import numpy as np
    if not REFERENCE.exists():
        raise FileNotFoundError(f'{REFERENCE} нет: выполните python tools/harvest_faces.py reference')
    with np.load(REFERENCE) as data:
        return {k: data[k] for k in data.files}


# --- reference / calibrate ---------------------------------------------------------------

def lolo_similarities(E, lectures):
    """cos каждого лица v1 к центроиду остальных лекций (leave-one-lecture-out)."""
    import numpy as np
    sims = np.empty(len(E), dtype=np.float32)
    for lecture in sorted(set(lectures.tolist())):
        inside = lectures == lecture
        sims[inside] = E[inside] @ unit(E[~inside].mean(0))
    return sims


def reference(args):
    import numpy as np
    from sokolov.data.download import image_info
    from sokolov.io import read_csv
    from .faces import FaceEngine, hamming
    engine = FaceEngine()
    rows = read_csv(V1 / 'manifest.csv')
    kept, missing, multiple, started = [], [], 0, time.perf_counter()
    for i, row in enumerate(rows, 1):
        image, _ = image_info(V1 / row['file'])
        faces = engine.detect(image)
        multiple += len(faces) > 1
        try:
            embedding, phash = engine.describe(image, faces[0]) if faces else (None, None)
        except ValueError:
            embedding = None
        if embedding is None:
            missing.append(row['file'])
        else:
            kept.append((row, faces[0], embedding, phash))
        if i % 200 == 0:
            print(f'reference {i}/{len(rows)}', flush=True)
    E = np.stack([k[2] for k in kept])
    lectures = np.array([k[0]['source_video'] for k in kept])
    centroid = unit(E.mean(0))
    lolo = lolo_similarities(E, lectures)
    per_lecture = {k: quantiles(lolo[lectures == k]) for k in sorted(set(lectures.tolist()))}
    S = E @ E.T
    np.fill_diagonal(S, -np.inf)
    same = lectures[:, None] == lectures[None, :]
    phashes = np.array([k[3] for k in kept], dtype=np.uint64)
    H = hamming(phashes, phashes)
    np.fill_diagonal(H, 64)
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REFERENCE, centroid=centroid, embeddings=E, phash=phashes, lectures=lectures,
             files=np.array([k[0]['file'] for k in kept]),
             timestamps=np.array([float(k[0]['timestamp_seconds']) for k in kept]),
             det_score=np.array([k[1]['confidence'] for k in kept], dtype=np.float32),
             face_px=np.array([k[1]['face_px'] for k in kept], dtype=np.float32),
             bbox=np.array([k[1]['bbox'] for k in kept], dtype=np.float32))
    by_lecture = Counter(r['source_video'] for r in rows)
    found = Counter(lectures.tolist())
    stats = {
        'created_utc': now(), 'images': len(rows), 'embedded': len(kept), 'missing': missing,
        'coverage': round(len(kept) / len(rows), 4), 'images_with_multiple_faces': int(multiple),
        'coverage_by_lecture': {k: f'{found[k]}/{v}' for k, v in sorted(by_lecture.items())},
        'similarity_to_leave_one_lecture_out_centroid': {'all': quantiles(lolo), **per_lecture},
        'similarity_to_full_centroid_in_sample': quantiles(E @ centroid),
        'nearest_neighbour_cos_same_lecture': quantiles(np.where(same, S, -np.inf).max(1)),
        'nearest_neighbour_cos_other_lecture': quantiles(np.where(~same, S, -np.inf).max(1)),
        'nearest_neighbour_phash_same_lecture': quantiles(np.where(same, H, 64).min(1)),
        'nearest_neighbour_phash_other_lecture': quantiles(np.where(~same, H, 64).min(1)),
        'det_score': quantiles([k[1]['confidence'] for k in kept]),
        'face_px': quantiles([k[1]['face_px'] for k in kept]),
        'models': engine.metadata, 'seconds': round(time.perf_counter() - started, 1),
        'note': 'Самое крупное лицо кадра; порог детектора из configs/arcface_face.toml.'}
    write_json(REFERENCE_STATS, stats)
    print(json.dumps({k: stats[k] for k in ('embedded', 'coverage', 'coverage_by_lecture',
                                            'similarity_to_leave_one_lecture_out_centroid')},
                     ensure_ascii=False, indent=1))
    print(REFERENCE, REFERENCE_STATS, sep='\n')
    return stats


def calibrate(args):
    """Пороги из LOLO-сходства Соколова (genuine) и лиц заведомо других людей (impostor)."""
    import numpy as np
    stats = json.loads(REFERENCE_STATS.read_text())
    genuine = stats['similarity_to_leave_one_lecture_out_centroid']['all']
    ref = load_reference()
    lolo = lolo_similarities(ref['embeddings'], ref['lectures'])
    rows = [r for path in args.negatives for r in read_rows(path) if r['review_status'] != 'excluded']
    if not rows:
        raise ValueError('Нет строк в --negatives')
    impostor_values = np.array([float(r['sokolov_similarity']) for r in rows])
    impostor = quantiles(impostor_values)
    per_source = {s: quantiles([float(r['sokolov_similarity']) for r in rows if r['source_id'] == s])
                  for s in sorted({r['source_id'] for r in rows})}
    margin = args.margin
    neg = round(impostor['max'] + margin, 2)
    pos = round(max(neg + 2 * margin, genuine['p5']), 2)
    solo = round((neg + pos) / 2, 2)
    flag = neg
    if not neg < solo < pos:
        raise ValueError(f'Распределения пересекаются: neg={neg}, pos={pos}, genuine p5={genuine["p5"]}; '
                         'нужно больше данных/ручной выбор')
    t = {'solo_min': solo, 'pos': pos, 'neg': neg, 'flag': flag,
         'near_duplicate': 0.97, 'v1_cos': 0.92, 'v1_phash': 6, 'v1_source_fraction': 0.25}
    effect = {name: {'genuine_lolo_below': round(float((lolo < value).mean()), 4),
                     'impostor_at_or_above': round(float((impostor_values >= value).mean()), 4)}
              for name, value in (('neg/flag', neg), ('solo_min', solo), ('pos', pos))}
    write_json(THRESHOLDS, {
        'thresholds': t, 'created_utc': now(),
        'similarity': 'cosine(ArcFace buffalo_l w600k_r50, L2-centroid 1000 v1 positives)',
        'rule': {
            'neg': f'max(impostor) + margin({margin})',
            'flag': '= neg: всё, что не уверенно «не Соколов», идёт на аудит',
            'pos': 'max(neg + 2·margin, genuine LOLO p5): ложная метка 1 хуже лишнего аудита',
            'solo_min': '(neg + pos) / 2: в solo-источнике априори Соколов',
            'v1_source_fraction': 'доля лиц источника с флагом dedup-v1 для предупреждения; по v1: '
                                  '0.53 для кадров той же лекции, 0.08 для другой (aligned pHash ≤ 6)',
            'near_duplicate': 'из задания; cos ≥ 0.97 к уже сохранённому лицу того же источника — пропуск',
            'v1_cos/v1_phash': 'из задания; dedup-v1 сверяет с basis.reference nearest-neighbour'},
        'basis': {
            'effect': effect,
            'genuine_lolo': genuine,
            'genuine_per_lecture': {k: v for k, v in stats['similarity_to_leave_one_lecture_out_centroid'].items()
                                    if k != 'all'},
            'impostor': impostor, 'impostor_by_source': per_source,
            'impostor_source_urls': {r['source_id']: r['source_url'] for r in rows},
            'impostor_manifests': [str(Path(p)) for p in args.negatives], 'impostor_notes': args.note,
            'reference_nn_cos_same_lecture': stats['nearest_neighbour_cos_same_lecture'],
            'reference_nn_cos_other_lecture': stats['nearest_neighbour_cos_other_lecture'],
            'reference_nn_phash_same_lecture': stats['nearest_neighbour_phash_same_lecture'],
            'reference_nn_phash_other_lecture': stats['nearest_neighbour_phash_other_lecture'],
            'reference_coverage': stats['coverage']},
        'caveats': ['genuine — только 4 лекции одной веб-камеры: в новых доменах (сцена, интервью, фото) '
                    'сходство Соколова ниже, поэтому solo/ambiguous обязательно смотреть глазами.',
                    'impostor — немного личностей; хвост распределения других людей оценён грубо.']})
    print(json.dumps(t, indent=1), THRESHOLDS, sep='\n')
    return t


# --- сбор ------------------------------------------------------------------------------

class SourceLock:
    """Один процесс на источник: неблокирующая flock-блокировка state/<source>.lock."""

    def __init__(self, layout, source_id):
        self.path = layout.state(source_id).with_suffix('.lock')

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+')
        try:
            fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.handle.close()
            raise RuntimeError(f'Источник {self.path.stem} уже обрабатывается другим процессом') from None
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.handle, fcntl.LOCK_UN)
        self.handle.close()


class Session:
    """Обработка кадров одного источника: детекция → эмбеддинг → метка → кроп → manifest."""

    def __init__(self, layout, source_id, args, thresholds, reference, engine):
        import numpy as np
        self.layout, self.source_id, self.args, self.t, self.engine = layout, source_id, args, thresholds, engine
        self.centroid = reference['centroid']
        self.prefix = f'{layout.prefix}_{source_id}_'
        existing = read_rows(layout.manifest)
        mine = [r for r in existing if r['source_id'] == source_id]
        self.keys = {natural_key(r) for r in mine}
        self.next = next_index(mine, self.prefix)
        self.excluded = {r['image_id'] for r in mine if r['review_status'] == 'excluded'}
        path = layout.embeddings(source_id)
        store = {}
        if path.exists():
            with np.load(path) as data:
                store = {k: data[k] for k in data.files}
        self.store = {'ids': list(store.get('ids', [])), 'emb': list(store.get('emb', [])),
                      'phash': list(store.get('phash', [])), 'bbox': list(store.get('bbox', [])),
                      'landmarks': list(store.get('landmarks', []))}
        self.counts = Counter()

    def known(self):
        import numpy as np
        rows = [e for i, e in zip(self.store['ids'], self.store['emb']) if i not in self.excluded]
        return np.stack(rows) if rows else np.zeros((0, 512), np.float32)

    def save_store(self):
        import numpy as np
        path = self.layout.embeddings(self.source_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.tmp.npz')
        np.savez(tmp, ids=np.array(self.store['ids'], dtype=str),
                 emb=np.array(self.store['emb'], dtype=np.float32).reshape(-1, 512),
                 phash=np.array(self.store['phash'], dtype=np.uint64),
                 bbox=np.array(self.store['bbox'], dtype=np.float32).reshape(-1, 4),
                 landmarks=np.array(self.store['landmarks'], dtype=np.float32).reshape(-1, 5, 2))
        tmp.replace(path)

    def add(self, image, timestamp, source_url):
        """Обработать кадр; вернуть добавленные строки manifest."""
        args, rows = self.args, []
        detected = self.engine.detect(image)
        faces = [f for f in detected if f['confidence'] >= args.min_det and f['face_px'] >= args.min_face]
        self.counts['faces_detected'] += len(detected)
        self.counts['skipped_low_det_or_small'] += len(detected) - len(faces)
        if args.max_faces:
            self.counts['skipped_max_faces'] += max(0, len(faces) - args.max_faces)
            faces = faces[:args.max_faces]
        known = self.known()
        for rank, face in enumerate(faces):
            try:
                embedding, phash = self.engine.describe(image, face)
            except ValueError:
                self.counts['skipped_alignment'] += 1
                continue
            if len(known) and float((known @ embedding).max()) >= self.t['near_duplicate']:
                self.counts['skipped_near_duplicate'] += 1
                continue
            sim = float(embedding @ self.centroid)
            label, status, note = classify(args.mode, sim, self.t)
            person, group = identity(label, self.source_id, args.person if args.mode == 'negative' else None,
                                     primary=rank == 0)
            row = {'label': label, 'source_id': self.source_id, 'group_id': group, 'domain': args.domain,
                   'person': person, 'source_url': source_url,
                   'timestamp_seconds': '' if timestamp is None else f'{timestamp:.3f}',
                   'det_score': f'{face["confidence"]:.4f}', 'face_px': str(int(round(face['face_px']))),
                   'sokolov_similarity': f'{sim:.4f}', 'review_status': status, 'notes': note}
            if natural_key(row) in self.keys:
                self.counts['skipped_already_in_manifest'] += 1
                continue
            row['image_id'] = f'{self.prefix}{self.next:04d}'
            self.next += 1
            path = self.layout.faces(self.source_id) / f'{row["image_id"]}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            crop_face(image, face['bbox'])[0].save(path, 'JPEG', quality=95)
            row['path'] = self.layout.rel(path)
            for key, value in (('ids', row['image_id']), ('emb', embedding), ('phash', phash),
                               ('bbox', face['bbox']), ('landmarks', face['landmarks'])):
                self.store[key].append(value)
            known = self.known()
            self.keys.add(natural_key(row))
            rows.append(row)
        if rows:
            self.save_store()
            appended = append_rows(self.layout.manifest, self.layout.lock, rows)
            if len(appended) != len(rows):
                raise RuntimeError('manifest изменился параллельно для этого источника')
            for row in rows:
                self.counts[f'saved_label{row["label"]}_{row["review_status"]}'] += 1
        return rows


def load_state(layout, source_id):
    path = layout.state(source_id)
    return json.loads(path.read_text()) if path.exists() else {'source_id': source_id, 'frames': {}, 'runs': []}


def run_frames(refs, fetch, session, state, layout, args):
    """Параллельная загрузка кадров (args.jobs потоков), последовательная обработка лиц.
    Останов: --max-seconds, max_failures подряд неудач, нет ни одного готового кадра stall s."""
    started, failures, done, ok, stop = time.monotonic(), 0, 0, 0, ''
    bytes0 = net.BYTES.value
    queue, inflight = list(refs), {}
    state_path = layout.state(session.source_id)
    pool = ThreadPoolExecutor(args.jobs)
    try:
        while queue or inflight:
            while queue and len(inflight) < 2 * args.jobs and not stop:
                ref = queue.pop(0)
                inflight[pool.submit(fetch, ref)] = (ref, time.monotonic())
            if not inflight:
                break
            finished, _ = wait(inflight, timeout=args.stall, return_when=FIRST_COMPLETED)
            if not finished:
                stop = f'нет ни одного готового кадра за {args.stall} s'
                for future in inflight:
                    future.cancel()
                break
            for future in finished:
                ref, submitted = inflight.pop(future)
                key, done = str(ref), done + 1
                try:
                    image, timestamp, source_url = future.result()
                except (net.FetchError, OSError, ValueError) as error:
                    failures += 1
                    state['frames'][key] = {'status': 'failed', 'error': str(error)[:300]}
                    print(f'[{done}/{len(refs)}] {key}: ОШИБКА {error}', flush=True)
                else:
                    rows = session.add(image, timestamp, source_url)
                    failures, ok = 0, ok + 1
                    state['frames'][key] = {'status': 'ok', 'time': timestamp, 'saved': [r['image_id'] for r in rows]}
                    summary = ', '.join(f'L{r["label"]} {r["review_status"][:3]} {float(r["sokolov_similarity"]):+.2f}'
                                        for r in rows)
                    print(f'[{done}/{len(refs)}] {key}: {image.width}×{image.height} px, сохранено {len(rows)}'
                          f'{" (" + summary + ")" if rows else ""}; {time.monotonic() - submitted:.1f} s', flush=True)
                write_json(state_path, state)
            if failures >= args.max_failures and not stop:
                stop = f'{failures} неудач подряд'
            if args.max_seconds and time.monotonic() - started > args.max_seconds and not stop:
                stop = f'превышен --max-seconds {args.max_seconds} s'
            if stop:
                queue.clear()
    finally:
        # Не ждём зависшие загрузки: у каждой свой таймаут (≤ ~3 мин), процесс дождётся их при выходе.
        pool.shutdown(wait=False, cancel_futures=True)
    seconds, media = time.monotonic() - started, net.BYTES.value - bytes0
    session.counts['phase_seconds'] = round(seconds, 1)
    session.counts['phase_bytes'] = media
    if ok:
        print(f'Фаза кадров: {ok} кадров за {seconds:.1f} s = {ok / seconds:.2f} кадр/s, {seconds / ok:.2f} s/кадр; '
              f'{media / 1024:.0f} KiB = {media / 1024 / ok:.0f} KiB/кадр, {media / 1024 / seconds:.0f} KiB/s', flush=True)
    return ok, done - ok, stop


def finish(layout, session, state, args, ok, failed, stop, started, bytes0, source_meta):
    seconds = time.monotonic() - started
    media = net.BYTES.value - bytes0
    run = {'finished_utc': now(), 'frames_ok': ok, 'frames_failed': failed, 'bytes': media,
           'seconds': round(seconds, 1), 'stopped': stop, 'counts': dict(session.counts),
           'args': {k: str(v) for k, v in vars(args).items() if k not in ('func',)}}
    state['runs'].append(run)
    write_json(layout.state(session.source_id), state)
    refresh_sources(layout, {session.source_id: {**source_meta, 'frames_sampled': ok, 'bytes_downloaded': media}})
    written = make_sheets(layout, session.source_id)
    print(f'Итог {session.source_id}: кадров {ok} ок, {failed} ошибок{"; ОСТАНОВ: " + stop if stop else ""}')
    print('  лица: ' + ', '.join(f'{k}={v}' for k, v in sorted(session.counts.items())))
    print(f'  скачано {media / 1024:.0f} KiB за {seconds:.1f} s; {ok / seconds if seconds else 0:.3f} кадр/s; '
          f'{media / 1024 / seconds if seconds else 0:.1f} KiB/s; {seconds / ok if ok else float("nan"):.2f} s/кадр')
    print('  лист: ' + ', '.join(str(p) for p in written))
    return run


def prepare_run(args):
    layout = Layout(args.dataset, Path(args.root))
    slug(args.source_id, 'source_id')
    if args.person:
        slug(args.person, 'person')
        if args.mode != 'negative' or args.person == 'sokolov':
            raise ValueError('--person — только для --mode negative (известный другой человек)')
    if args.dataset == 'web_negative_v2' and args.mode != 'negative':
        raise ValueError('web_negative_v2 собирается только в --mode negative')
    if args.dataset == 'web_positive_v2' and args.mode == 'negative':
        raise ValueError('web_positive_v2: режимы solo или multi')
    return layout, load_thresholds(), load_reference()


def video(args):
    from .faces import FaceEngine
    layout, thresholds, ref = prepare_run(args)
    net.install_dns_fallback()
    with SourceLock(layout, args.source_id):
        started, bytes0 = time.monotonic(), net.BYTES.value
        info, extra = net.resolve(args.url, shlex.split(args.ytdlp_args or ''))
        fmt = net.pick_format(info, args.max_height, args.max_kbps)
        headers = {**(info.get('http_headers') or {}), **(fmt.get('http_headers') or {})}
        duration = info.get('duration')
        estimate = net.estimate_bytes(fmt, duration)
        strategy = args.fetch
        if strategy == 'auto':
            strategy = ('download' if estimate and estimate <= args.download_max_mib << 20
                        else 'hls' if fmt.get('protocol', '').startswith('m3u8') else 'seek')
        print(f'{info.get("extractor_key")}: «{info.get("title")}», {duration} s; формат {fmt.get("format_id")} '
              f'{fmt.get("height")}p {fmt.get("protocol")} ~{(estimate or 0) >> 20} MiB целиком; '
              f'стратегия {strategy}; resolve {info.get("_resolve_seconds")} s', flush=True)
        downloaded, holder, lock = None, {'generation': 0, 'resolves': 0}, threading.Lock()

        def build(reresolve=False):
            """Новый resolve даёт другой узел CDN: часть узлов отсюда недоступна (connect timeout)."""
            nonlocal info, fmt, headers
            if reresolve:
                holder['resolves'] += 1
                info, _ = net.resolve(args.url, extra)
                fmt = net.pick_format(info, args.max_height, args.max_kbps)
                headers = {**(info.get('http_headers') or {}), **(fmt.get('http_headers') or {})}
            if strategy == 'hls':
                hls = net.HLS(fmt['url'], headers, args.max_height)
                holder['duration'] = duration or hls.duration
                holder['fetch'] = hls.frame
            else:
                url, hdrs = fmt['url'], headers
                holder['fetch'] = lambda t: net.ffmpeg_frame(url, t, hdrs, args.exact)[:2]
            holder['generation'] += 1

        def build_with_retries():
            for attempt in range(args.max_resolves + 1):
                try:
                    return build(reresolve=attempt > 0)
                except net.FetchError as error:
                    if attempt == args.max_resolves:
                        raise
                    print(f'  {error}; повторный resolve {attempt + 1}/{args.max_resolves}', flush=True)

        def fetch(t):
            generation = holder['generation']
            try:
                return (*holder['fetch'](t), args.url)
            except net.FetchError as error:
                if strategy == 'download':
                    raise
                with lock:  # одна пересборка на поколение, сколько бы потоков ни упало одновременно
                    if holder['generation'] == generation:
                        if holder['resolves'] >= args.max_resolves:
                            raise
                        print(f'  {error}; повторный resolve {holder["resolves"] + 1}/{args.max_resolves}',
                              flush=True)
                        build(reresolve=True)
                return (*holder['fetch'](t), args.url)
        try:
            state = load_state(layout, args.source_id)
            todo, total = None, duration
            for attempt in range(2):  # длительность из yt-dlp; иначе — из HLS-плейлиста
                if total or args.end is not None:
                    refs = timestamps(total or args.end, args.start, args.end, args.count, args.every)
                    todo = [t for t in refs if state['frames'].get(f'{t:.3f}', {}).get('status') != 'ok']
                    print(f'Кадров: {len(refs)}, уже обработано {len(refs) - len(todo)}, к загрузке {len(todo)}',
                          flush=True)
                if todo == [] or 'fetch' in holder:
                    break
                if strategy == 'download':
                    downloaded = net.download(args.url, fmt['format_id'], SCRATCH / 'downloads' / args.source_id,
                                              extra)
                    holder['fetch'] = lambda t: net.ffmpeg_frame(downloaded, t, exact=args.exact)[:2]
                else:
                    build_with_retries()
                total = total or holder.get('duration')
                if todo is not None:
                    break
            if todo is None:
                raise ValueError('Длительность неизвестна — задайте --end')
            state.update(url=args.url, format=fmt.get('format_id'), strategy=strategy)
            session = Session(layout, args.source_id, args, thresholds, ref, FaceEngine())
            ok, failed, stop = run_frames([f'{t:.3f}' for t in todo], lambda key: fetch(float(key)),
                                          session, state, layout, args)
        finally:
            if downloaded:
                Path(downloaded).unlink(missing_ok=True)
        date = info.get('upload_date') or ''
        meta = {'url': args.url, 'platform': info.get('extractor_key') or '', 'title': info.get('title') or '',
                'date': f'{date[:4]}-{date[4:6]}-{date[6:]}' if len(date) == 8 else '',
                'duration_s': duration or total, 'domain': args.domain,
                'persons': [args.person] if args.person else [],
                'identity_verification': args.verification or ''}
        return finish(layout, session, state, args, ok, failed, stop, started, bytes0, meta)


def images(args):
    from io import BytesIO
    from PIL import Image, ImageOps
    from .faces import FaceEngine
    layout, thresholds, ref = prepare_run(args)
    if args.dir and not args.source_url:
        raise ValueError('--dir требует --source-url (страница, откуда взяты фото)')
    net.install_dns_fallback()
    with SourceLock(layout, args.source_id):
        started, bytes0 = time.monotonic(), net.BYTES.value
        if args.dir:
            files = sorted(p for p in Path(args.dir).iterdir()
                           if p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp') and p.is_file())
            refs = [f'file:{p.name}' for p in files]
        else:
            refs = list(args.urls)
        if not refs:
            raise ValueError('Нет изображений')
        state = load_state(layout, args.source_id)
        todo = [r for r in refs if state['frames'].get(r, {}).get('status') != 'ok']

        def fetch(ref):
            if ref.startswith('file:'):
                data, url = (Path(args.dir) / ref[5:]).read_bytes(), args.source_url
            else:
                data, url = net.http_get(ref, timeout=15, attempts=4, deadline=90, max_bytes=25 << 20), ref
            try:
                with Image.open(BytesIO(data)) as source:
                    image = ImageOps.exif_transpose(source).convert('RGB')
            except OSError as error:
                raise net.FetchError(f'не изображение: {error}') from None
            return image, None, url

        print(f'Изображений: {len(refs)}, к обработке {len(todo)}', flush=True)
        session = Session(layout, args.source_id, args, thresholds, ref, FaceEngine())
        ok, failed, stop = run_frames(todo, fetch, session, state, layout, args)
        meta = {'url': args.source_url or (args.urls[0] if args.urls else ''),
                'platform': 'local' if args.dir else 'web', 'title': args.title or '', 'date': args.date or '',
                'duration_s': '', 'domain': args.domain, 'persons': [args.person] if args.person else [],
                'identity_verification': args.verification or ''}
        return finish(layout, session, state, args, ok, failed, stop, started, bytes0, meta)


# --- аудит ----------------------------------------------------------------------------

def make_sheets(layout, source_id=None, order='time', statuses=None):
    rows = [r for r in read_rows(layout.manifest) if (source_id is None or r['source_id'] == source_id)
            and (not statuses or r['review_status'] in statuses)]
    written = []
    for source in sorted({r['source_id'] for r in rows} or ([source_id] if source_id else [])):
        mine = sheets.sort_rows([r for r in rows if r['source_id'] == source], order)
        name = source + (f'__{"-".join(statuses)}' if statuses else '') + ('__sim' if order == 'sim' else '')
        counts = Counter(f'L{r["label"]}/{r["review_status"]}' for r in mine)
        title = f'{layout.dataset}/{source}: {len(mine)} faces ' + ' '.join(f'{k}={v}' for k, v in sorted(counts.items()))
        written += sheets.render(layout.root, mine, layout.sheet(name), title)
    return written


def sheet(args):
    layout = Layout(args.dataset, Path(args.root))
    for path in make_sheets(layout, args.source, args.sort, args.status):
        print(path)


def dedup_v1(args):
    import numpy as np
    from .faces import hamming
    layout = Layout(args.dataset, Path(args.root))
    t, ref = load_thresholds(), load_reference()
    rows = [r for r in read_rows(layout.manifest) if r['review_status'] == 'accepted']
    flagged = {}
    for source in sorted({r['source_id'] for r in rows}):
        path = layout.embeddings(source)
        if not path.exists():
            print(f'{source}: нет {path}', flush=True)
            continue
        with np.load(path) as store:
            index = {i: k for k, i in enumerate(store['ids'].tolist())}
            emb, phash = store['emb'], store['phash']
        mine = [r for r in rows if r['source_id'] == source and r['image_id'] in index]
        if not mine:
            continue
        k = [index[r['image_id']] for r in mine]
        cos = emb[k] @ ref['embeddings'].T
        dist = hamming(phash[k], ref['phash'])
        hits = 0
        for row, c, d in zip(mine, cos, dist):
            j, jd = int(c.argmax()), int(d.argmin())
            if c[j] >= t['v1_cos'] or d[jd] <= t['v1_phash']:
                hits += 1
                flagged[row['image_id']] = (f'dup-v1: cos={c[j]:.3f} ({ref["files"][j]}), '
                                            f'phash={int(d[jd])} ({ref["files"][jd]})')
        # По v1: доля лиц с совпадением ≈ 53 % для кадров той же лекции и ≈ 8 % для другой лекции.
        suspicious = len(mine) >= 10 and hits / len(mine) >= t.get('v1_source_fraction', .25)
        print(f'{source}: accepted {len(mine)}, похожи на v1 {hits} ({hits / len(mine):.0%}); '
              f'max cos {cos.max():.3f}, min pHash {int(dist.min())}'
              + ('  <-- вероятно, та же запись, что лекция v1: проверить источник целиком' if suspicious else ''))
    if args.dry_run or not flagged:
        print(f'Отмечено бы: {len(flagged)}' if args.dry_run else 'Совпадений с v1 нет')
        return flagged

    def change(all_rows):
        for row in all_rows:
            if row['image_id'] in flagged and row['review_status'] == 'accepted':
                row['review_status'], row['notes'] = 'ambiguous', flagged[row['image_id']]
        return len(flagged)
    update_rows(layout.manifest, layout.lock, change)
    refresh_sources(layout)
    print(f'Переведено в ambiguous: {len(flagged)}')
    return flagged


def set_status(args):
    layout = Layout(args.dataset, Path(args.root))
    ids = set(args.ids)
    if args.person:
        slug(args.person, 'person')

    def change(rows):
        found = {r['image_id'] for r in rows} & ids
        if found != ids:
            raise ValueError(f'Нет в manifest: {sorted(ids - found)[:10]}')
        for row in rows:
            if row['image_id'] in ids:
                row['review_status'], row['notes'] = args.status, f'audit: {args.note}'
                if args.label is not None:
                    row['label'] = str(args.label)
                    row['person'], row['group_id'] = identity(args.label, row['source_id'], args.person)
                elif args.person:
                    row['person'], row['group_id'] = identity(row['label'], row['source_id'], args.person)
        return len(ids)
    print(f'Обновлено строк: {update_rows(layout.manifest, layout.lock, change)}')
    refresh_sources(layout)


def stats(args):
    layout = Layout(args.dataset, Path(args.root))
    rows = read_rows(layout.manifest)
    sources = {s['source_id']: s for s in (json.loads(layout.sources.read_text()) if layout.sources.exists() else [])}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['source_id']].append(row)
    report = {}
    for source in sorted(set(grouped) | set(sources)):
        mine = grouped.get(source, [])
        meta = sources.get(source, {})
        report[source] = {
            'counts': dict(sorted(Counter(f'L{r["label"]}/{r["review_status"]}' for r in mine).items())),
            'sim_accepted_label1': quantiles([float(r['sokolov_similarity']) for r in mine
                                              if r['review_status'] == 'accepted' and r['label'] == '1']),
            'sim_accepted_label0': quantiles([float(r['sokolov_similarity']) for r in mine
                                              if r['review_status'] == 'accepted' and r['label'] == '0']),
            'frames_sampled': meta.get('frames_sampled'), 'bytes_downloaded': meta.get('bytes_downloaded'),
            'domain': meta.get('domain'), 'persons': meta.get('persons')}
    total = Counter(f'L{r["label"]}/{r["review_status"]}' for r in rows)
    print(json.dumps({'dataset': layout.dataset, 'rows': len(rows), 'total': dict(sorted(total.items())),
                      'bytes_downloaded': sum(int(s.get('bytes_downloaded') or 0) for s in sources.values()),
                      'sources': report}, ensure_ascii=False, indent=1))
    return report
