"""Independent checks of generated images: transfer score, SCRFD faces, ArcFace identity.

Every unavailable check is reported as N/A with a reason; nothing is downloaded here.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time

from .io import read_csv, read_json, sha256, write_csv, write_json

CLASSES = ('face-like', 'many faces', 'pattern')
CLASS_RU = {'face-like': 'одно лицо', 'many faces': 'несколько лиц', 'pattern': 'узор (0 лиц)', 'N/A': 'N/A'}
CANDIDATE_THRESHOLD = .05  # SCRFD candidates kept only to report max confidence below the face threshold
REAL_PER_CLASS = 10        # real test crops per class used as a reference
MAX_REFERENCE = 2000       # train positives averaged into the ArcFace centroid
REFERENCE_NPZ = Path('data/reference/sokolov_reference.npz')
PADDING_NOTE = ('image centred on a black 2S x 2S canvas: SCRFD misses faces that fill the whole frame of a '
                'tightly aligned crop; the canvas then follows the training detector input policy '
                '(shrink_only_top_left_pad_v1: no enlargement, black padding to 640 px)')


def face_class(faces):
    if faces is None:
        return 'N/A'
    return CLASSES[0] if faces == 1 else CLASSES[1] if faces > 1 else CLASSES[2]


def _tv(array):
    import numpy as np
    x = array.astype(np.float64)/255
    return float(np.abs(np.diff(x, axis=0)).mean()+np.abs(np.diff(x, axis=1)).mean())


class FaceTools:
    """SCRFD det_10g + ArcFace w600k_r50 from the local buffalo_l pack (CPU, never downloads)."""

    def __init__(self, data):
        folder = Path(data.detector_root)/'models'/data.detector_name
        detector, recognizer = folder/'det_10g.onnx', folder/'w600k_r50.onnx'
        missing = [p.name for p in (detector, recognizer) if not p.is_file()]
        if missing:
            raise FileNotFoundError(f'нет файлов {", ".join(missing)} в {folder}')
        from insightface.model_zoo import get_model
        self.detector = get_model(str(detector), providers=['CPUExecutionProvider'])
        self.detector.prepare(ctx_id=-1, input_size=(640, 640), det_thresh=CANDIDATE_THRESHOLD)
        self.recognizer = get_model(str(recognizer), providers=['CPUExecutionProvider'])
        self.recognizer.prepare(ctx_id=-1)
        self.threshold = data.detector_threshold
        from .data.prepare import DETECTOR_RESIZE_POLICY
        self.metadata = {'detector': 'SCRFD det_10g.onnx (insightface)', 'detector_sha256': sha256(detector),
            'recognizer': 'ArcFace w600k_r50.onnx (insightface)', 'recognizer_sha256': sha256(recognizer),
            'threshold': self.threshold, 'candidate_threshold': CANDIDATE_THRESHOLD,
            'input_size': [640, 640], 'resize_policy': DETECTOR_RESIZE_POLICY, 'padding': PADDING_NOTE,
            'provider': 'CPUExecutionProvider'}

    def embed(self, bgr_images):
        import numpy as np
        features = np.concatenate([self.recognizer.get_feat(bgr_images[i:i+32])
                                   for i in range(0, len(bgr_images), 32)])
        return features/np.linalg.norm(features, axis=1, keepdims=True)

    def inspect(self, rgb, centroid=None):
        """Faces >= threshold, max candidate confidence and best ArcFace cosine to the centroid."""
        import numpy as np
        from insightface.utils import face_align
        from .data.prepare import detector_input
        height, width = rgb.shape[:2]
        canvas = np.zeros((2*height, 2*width, 3), np.uint8)
        canvas[height//2:height//2+height, width//2:width//2+width] = rgb
        bgr = canvas[:, :, ::-1].copy()
        padded, scale = detector_input(canvas)
        boxes, landmarks = self.detector.detect(padded, max_num=0)
        if len(boxes):
            landmarks = landmarks/scale
        confident = [i for i in range(len(boxes)) if boxes[i, 4] >= self.threshold]
        cosine = None
        if confident and centroid is not None:
            crops = [face_align.norm_crop(bgr, landmark=landmarks[i], image_size=112) for i in confident]
            cosine = float((self.embed(crops)@centroid).max())
        return {'faces': len(confident), 'max_confidence': float(boxes[:, 4].max()) if len(boxes) else None,
                'arcface_cosine': cosine}


def _manifest_rows(data):
    path = Path(data.prepared_manifest)
    if not path.is_file():
        return None, f'нет подготовленного манифеста {path}'
    rows = read_csv(path)
    needed = {'split', 'label', 'status', 'crop_path'}
    if not rows or not needed <= set(rows[0]):
        return None, f'в манифесте нет колонок {sorted(needed)}'
    return rows, ''


def _load_crop(row, size=None):
    import numpy as np
    from PIL import Image
    path = Path(row['crop_path'])
    if not path.is_file() or (row.get('crop_sha256') and sha256(path) != row['crop_sha256']):
        raise ValueError(f'кроп отсутствует или изменён: {path.name}')
    with Image.open(path) as image:
        image = image.convert('RGB')
        if size:
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image)


def reference_centroid(data, tools):
    """Mean of L2-normalised ArcFace embeddings of the run's train positives (aligned 224 crops -> 112)."""
    import numpy as np
    rows, reason = _manifest_rows(data)
    if rows is not None:
        chosen = sorted((r for r in rows if r['split'] == 'train' and r['label'] == '1' and r['status'] == 'ok'),
                        key=lambda r: r.get('crop_sha256') or r['crop_path'])[:MAX_REFERENCE]
        try:
            images = [_load_crop(r, 112)[:, :, ::-1].copy() for r in chosen]
        except (ValueError, OSError) as error:
            images, reason = [], str(error)
        if images:
            mean = tools.embed(images).mean(0)
            return mean/np.linalg.norm(mean), {'available': True, 'reason': '', 'count': len(images),
                'source': 'train positives of the prepared manifest', 'manifest_sha256': sha256(data.prepared_manifest)}
        reason = reason or 'в train нет положительных кропов'
    if REFERENCE_NPZ.is_file():
        stored = np.load(REFERENCE_NPZ)
        vectors = stored['centroid'][None] if 'centroid' in stored else stored['embeddings'] if 'embeddings' in stored else None
        if vectors is not None and vectors.ndim == 2 and vectors.shape[1] == 512:
            vectors = vectors/np.linalg.norm(vectors, axis=1, keepdims=True)
            mean = vectors.mean(0)
            return mean/np.linalg.norm(mean), {'available': True, 'reason': f'manifest: {reason}',
                'count': int(len(vectors)), 'source': str(REFERENCE_NPZ), 'manifest_sha256': sha256(REFERENCE_NPZ)}
    return None, {'available': False, 'reason': reason, 'count': 0, 'source': None}


class Scorer:
    """CPU classifier restored from a checkpoint; scores uint8 RGB images resized like the dataset."""

    def __init__(self, checkpoint):
        import torch
        from .model import build_model
        from .synthesize import checkpoint_config
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        config = checkpoint_config(saved)
        self.model = build_model(config.model, pretrained=False)
        self.model.load_state_dict(saved['model'], strict=True)
        self.model.eval().requires_grad_(False)
        self.architecture, self.size = config.model.architecture, config.model.image_size
        self.policy = config.model.face_preprocessing
        self.data_fingerprint = saved.get('data_fingerprint')
        self.sha256 = sha256(checkpoint)

    def logit(self, rgb):
        import numpy as np
        import torch
        from PIL import Image
        from .synthesize import normalized_pixels
        if rgb.shape[:2] != (self.size, self.size):
            rgb = np.asarray(Image.fromarray(rgb).resize((self.size, self.size), Image.Resampling.BILINEAR))
        x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()[None]/255
        with torch.no_grad():
            return float(self.model(normalized_pixels(x, self.architecture, self.policy)).reshape(-1)[0])


def _sigmoid(value):
    import math
    return 1/(1+math.exp(-value)) if value >= 0 else math.exp(value)/(1+math.exp(value))


def _stats(items, key):
    values = [r[key] for r in items if r.get(key) is not None]
    if not values:
        return None
    return {'mean': sum(values)/len(values), 'min': min(values), 'max': max(values), 'n': len(values)}


def aggregate(items):
    classes = Counter(r['class'] for r in items)
    return {'n': len(items), **{key: _stats(items, key) for key in
            ('score', 'logit', 'tv', 'transfer_score', 'max_confidence', 'arcface_cosine')},
            'classes': {c: classes.get(c, 0) for c in (*CLASSES, 'N/A')}}


def analyze_experiment(runs, root, *, config, checkpoint, run_dir=None):
    """Score finals/initial noises and a few real test crops with independent tools."""
    import numpy as np
    from PIL import Image
    started = time.perf_counter()
    root = Path(root)
    result = {'schema_version': 1}
    own = Scorer(checkpoint)
    transfer, transfer_info = None, {'available': False, 'reason': 'не задан каталог запуска с моделями'}
    if run_dir is not None:
        other = 'arcface' if config.model.architecture == 'resnet18' else 'resnet18'
        paths = {name: Path(run_dir)/name/'best.pt' for name in ('resnet18', 'arcface')}
        if all(p.is_file() for p in paths.values()):
            transfer = Scorer(paths[other])
            transfer_info = {'available': True, 'reason': '', 'architecture': other,
                             'checkpoint_sha256': transfer.sha256, 'image_size': transfer.size,
                             'resize': 'PIL bilinear from the generated size, as dataset preprocessing'}
            if own.data_fingerprint != transfer.data_fingerprint:
                transfer, transfer_info = None, {'available': False, 'reason': 'модели обучены на разных данных'}
        else:
            transfer_info['reason'] = 'в запуске нет обоих best.pt (resnet18 и arcface)'
    result['transfer'] = transfer_info
    try:
        tools = FaceTools(config.data)
    except (FileNotFoundError, ImportError, OSError, RuntimeError) as error:
        tools = None
        result['detector'] = {'available': False, 'reason': f'{type(error).__name__}: {error}'}
    else:
        result['detector'] = {'available': True, 'reason': '', **tools.metadata}
    centroid, result['reference'] = (reference_centroid(config.data, tools) if tools else
        (None, {'available': False, 'reason': 'детектор/ArcFace недоступны', 'count': 0, 'source': None}))

    def measure(rgb):
        record = {'transfer_score': None, 'faces': None, 'max_confidence': None, 'arcface_cosine': None}
        if transfer is not None:
            record['transfer_score'] = _sigmoid(transfer.logit(rgb))
        if tools is not None:
            record.update(tools.inspect(rgb, centroid))
        record['class'] = face_class(record['faces'])
        return record

    images = []
    for run in runs:
        trajectory = read_csv(run.trajectory)
        for kind, row, path in (('initial', trajectory[0], run.snapshots_dir/'step_0000.png'),
                                ('final', trajectory[-1], run.final_image)):
            with Image.open(path) as image:
                rgb = np.asarray(image.convert('RGB'))
            images.append({'kind': kind, 'variant': run.variant, 'seed': run.seed, 'score': float(row['score']),
                           'logit': float(row['logit']), 'tv': float(row['tv']), **measure(rgb)})
    result['images'] = images
    result['per_variant'] = {name: {kind: aggregate([r for r in images if r['variant'] == name and r['kind'] == kind])
                                    for kind in ('final', 'initial')}
                             for name in dict.fromkeys(r.variant for r in runs)}
    # Real test crops: only aggregates leave this function (no identities or paths in reports).
    real = {'available': False, 'reason': '', 'per_class': REAL_PER_CLASS}
    rows, reason = _manifest_rows(config.data)
    if rows is None:
        real['reason'] = reason
    else:
        for label, name in (('1', 'positive'), ('0', 'negative')):
            chosen = sorted((r for r in rows if r['split'] == 'test' and r['label'] == label and r['status'] == 'ok'),
                            key=lambda r: hashlib.sha256(r.get('image_id', r['crop_path']).encode()).hexdigest())
            records = []
            for row in chosen[:REAL_PER_CLASS]:
                try:
                    rgb = _load_crop(row, own.size)
                except (ValueError, OSError) as error:
                    real['reason'] = str(error)
                    continue
                logit = own.logit(rgb)
                records.append({'score': _sigmoid(logit), 'logit': logit, 'tv': _tv(rgb), **measure(rgb)})
            real[name] = aggregate(records)
        real['available'] = bool(real['positive']['n'] or real['negative']['n'])
        real['reason'] = real['reason'] or ('' if real['available'] else 'в test нет пригодных кропов')
        real['preprocessing'] = ('aligned 224 px crop resized to the model input with PIL bilinear, then the '
                                 f'checkpoint face preprocessing ({own.policy})')
    result['real'] = real
    result['seconds'] = time.perf_counter()-started
    write_json(root/'analysis.json', result)
    write_csv(root/'analysis.csv', images, fields=['kind', 'variant', 'seed', 'score', 'logit', 'tv', 'transfer_score',
              'faces', 'max_confidence', 'arcface_cosine', 'class'])
    return result


def _fmt(value, digits=4):
    return 'N/A' if value is None else f'{value:.{digits}f}'


def _mean(stats, digits=4):
    return 'N/A' if not stats else _fmt(stats['mean'], digits)


def _confidence(value, detector_available):
    """SCRFD keeps candidates >= CANDIDATE_THRESHOLD only; None then means 'below it'."""
    return 'N/A' if not detector_available else f'<{CANDIDATE_THRESHOLD:g}' if value is None else f'{value:.3f}'


def interpretation(analysis, results):
    """Sentences built only from measured values; no conclusion is pre-written."""
    lines = []
    finals = {name: groups['final'] for name, groups in analysis['per_variant'].items()}
    initials = {name: groups['initial'] for name, groups in analysis['per_variant'].items()}
    for name, final in finals.items():
        n = final['n']
        line = (f"{name}: финальная оценка в среднем {_mean(final['score'])} "
                f"(мин {_fmt(final['score']['min'])}, макс {_fmt(final['score']['max'])}); "
                f"TV {_mean(final['tv'], 3)} против {_mean(initials[name]['tv'], 3)} у начального изображения")
        if analysis['detector']['available']:
            c = final['classes']
            line += (f"; SCRFD: одно лицо {c['face-like']}/{n}, несколько {c['many faces']}/{n}, "
                     f"узор {c['pattern']}/{n}, максимальная уверенность "
                     f"{_confidence((final['max_confidence'] or {}).get('max'), True)}")
            line += (f"; ArcFace-косинус найденных лиц {_mean(final['arcface_cosine'], 3)}"
                     if final['arcface_cosine'] else '; лиц для ArcFace нет')
        if analysis['transfer']['available']:
            line += f"; оценка второй моделью ({analysis['transfer']['architecture']}) {_mean(final['transfer_score'])}"
        lines.append(line+'.')
    best = max(finals.values(), key=lambda f: f['score']['mean'])['score']['mean']
    lines.append('Наибольшая средняя финальная оценка: ' +
                 ', '.join(k for k, f in finals.items() if f['score']['mean'] == best) + f' ({_fmt(best)}).')
    if analysis['detector']['available']:
        most = max(f['classes']['face-like'] for f in finals.values())
        lines.append('Ни в одном финале детектор не нашёл ровно одно лицо.' if most == 0 else
                     'Больше всего финалов ровно с одним лицом: ' +
                     ', '.join(k for k, f in finals.items() if f['classes']['face-like'] == most) + f' ({most}).')
    else:
        lines.append(f"Проверка лиц: N/A — {analysis['detector']['reason']}.")
    real = analysis['real']
    if real['available']:
        for name in ('positive', 'negative'):
            group = real[name]
            label = 'позитивы' if name == 'positive' else 'негативы'
            text = f"Реальные test-{label} (n={group['n']}): оценка {_mean(group['score'])}"
            if analysis['detector']['available']:
                text += (f", одно лицо {group['classes']['face-like']}/{group['n']}, "
                         f"ArcFace-косинус {_mean(group['arcface_cosine'], 3)}")
            if analysis['transfer']['available']:
                text += f", оценка второй моделью {_mean(group['transfer_score'])}"
            lines.append(text+'.')
        positive, negative = real['positive'].get('arcface_cosine'), real['negative'].get('arcface_cosine')
        if positive and negative:
            for name, final in finals.items():
                value = final['arcface_cosine']
                if not value:
                    continue
                where = ('не ниже среднего реальных позитивов' if value['mean'] >= positive['mean'] else
                         'между средними негативов и позитивов' if value['mean'] > negative['mean'] else
                         'не выше среднего реальных негативов')
                lines.append(f"{name}: средний ArcFace-косинус найденных лиц {value['mean']:.3f} — {where} "
                             f"({negative['mean']:.3f} / {positive['mean']:.3f}).")
    else:
        lines.append(f"Сравнение с реальными кропами: N/A — {real['reason']}.")
    if not all(r['weights_unchanged'] for r in results):
        lines.append('ВНИМАНИЕ: веса классификатора изменились хотя бы в одном запуске.')
    return lines


def publication_files(summary):
    if 'variants' not in summary:  # experiment written before generation variants (e.g. local-mps-v2)
        return {'synthesis/summary.json', 'synthesis/seed_comparison.png', 'synthesis/scores.png'} | {
            f'synthesis/seed_{seed}/{name}' for seed in summary['seeds'] for name in ('progression.png', 'trajectory.csv')}
    files = {'synthesis/summary.json', 'synthesis/grid.png', 'synthesis/scores.png',
             'synthesis/trajectories.csv', 'synthesis/analysis.csv'}
    return files | {f'synthesis/seed_{seed}/progression.png' for seed in summary['seeds']}


def legacy_report_section(source, destination, summary):
    """Section for a single-baseline experiment written before generation variants."""
    lines = ['', '## Генерация из случайного шума', '',
             'Веса выбранной модели фиксированы. Adam изменяет только пиксели входной картинки.',
             'Максимизируется логит «Соколов» с TV- и L2-регуляризацией. После каждого шага пиксели ограничены диапазоном [0,1].',
             'Оценки на графике рассчитаны по сохранённым 8-битным пикселям. Детектор лиц в оптимизации не участвует.',
             'Высокая оценка не гарантирует узнаваемое лицо. Показаны все начальные шумы; рост оценки не обязан быть монотонным.', '',
             '```json', json.dumps(summary, ensure_ascii=False, indent=2), '```', '']
    for image in ('seed_comparison.png', 'scores.png'):
        shutil.copy2(source/image, destination/image)
        lines.append(f'![Synthesis {image}](synthesis/{image})')
    for seed in summary['seeds']:
        target = destination/f'seed_{seed}'
        target.mkdir(exist_ok=True)
        for name in ('progression.png', 'trajectory.csv'):
            shutil.copy2(source/f'seed_{seed}'/name, target/name)
        lines += ['', f'### Начальный шум {seed}', '', f'![Steps for seed {seed}](synthesis/seed_{seed}/progression.png)', '']
    return lines


def report_section(run, destination, summary):
    """Copy publication-safe synthetic artifacts and return the Russian Markdown section."""
    run, destination = Path(run), Path(destination)
    source = run/'synthesis'
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination/'summary.json', summary)
    if 'variants' not in summary:
        return legacy_report_section(source, destination, summary)
    for name in ('grid.png', 'scores.png', 'trajectories.csv', 'analysis.csv'):
        shutil.copy2(source/name, destination/name)
    for seed in summary['seeds']:
        (destination/f'seed_{seed}').mkdir(exist_ok=True)
        shutil.copy2(source/'by_seed'/f'seed_{seed}.png', destination/f'seed_{seed}'/'progression.png')
    analysis = summary['analysis']
    generation = summary['generation']
    lines = ['', '## Генерация из случайного шума', '',
             f"Модель: {summary['architecture']} ({summary['image_size']}×{summary['image_size']} px), "
             'веса зафиксированы (eval, градиент только по изображению); хеш весов до/после совпадает в каждом запуске.',
             f"Перед классификатором применяется та же дифференцируемая обработка лица, что при обучении и в UI: "
             f"{summary.get('face_preprocessing', 'rgb')}.",
             f"Шагов оптимизации: {generation['steps']}, снимки каждые {generation['snapshot_every']} шагов, "
             f"начальные шумы seed {', '.join(map(str, summary['seeds']))}.",
             'Оценка на кривых и в таблицах вычислена по сохранённым 8-битным PNG без случайных преобразований. '
             'Детектор и ArcFace в оптимизации не участвуют и служат независимой проверкой.', '',
             '### Варианты ограничений', '',
             '| Вариант | Параметризация и ограничения |', '|---|---|']
    lines += [f"| {name} | {text} |" for name, text in summary['variant_descriptions'].items()]
    lines += ['', 'pixel — исходная постановка (сырые пиксели, слабые TV/L2). robust — те же пиксели, но классификатор '
              'на каждом шаге видит случайно сдвинутую/масштабированную/повёрнутую копию, параметр периодически '
              'размывается, TV сильнее. fourier — изображение задаётся спектром с масштабом 1/f и декоррелированным '
              'цветом (Olah et al., Distill 2017, «Feature Visualization»), затем те же преобразования. '
              'У fourier начальное изображение — почти однотонное поле из случайного спектра, а не равномерный шум.', '',
              '| Вариант | Время оптимизации, с (сумма по seed) | Среднее на запуск, с | Устройство |', '|---|---|---|---|']
    for name, value in summary['runtime'].items():
        lines.append(f"| {name} | {value['seconds_total']:.1f} | {value['seconds_per_run_mean']:.1f} | "
                     f"{', '.join(value['devices'])} |")
    lines += ['', '![Финальные изображения: вариант × seed](synthesis/grid.png)', '',
              '![Оценка и логит по шагам для каждого варианта](synthesis/scores.png)', '',
              '### Анализ финальных изображений', '']
    detector, reference, transfer = analysis['detector'], analysis['reference'], analysis['transfer']
    lines.append('Лица: ' + (f"SCRFD det_10g, порог {detector['threshold']:g}; изображение помещается в центр чёрного "
                 'холста 2S×2S, иначе детектор не находит лицо, занимающее весь кадр выровненного кропа. '
                 'Класс: одно лицо / несколько лиц / узор (0 лиц).' if detector['available'] else f"N/A — {detector['reason']}."))
    lines.append('ArcFace-косинус: w600k_r50 (независимая от классификатора модель), максимум по найденным лицам; '
                 + (f"центроид — {reference['count']} кропов ({reference['source']})." if reference['available']
                    else f"N/A — {reference['reason']}."))
    lines.append('Перенос: ' + (f"оценка второй обученной моделью ({transfer['architecture']}), изображение "
                 'масштабировано билинейно.' if transfer['available'] else f"N/A — {transfer['reason']}."))
    lines += ['', '| Вариант | Изображения | n | Оценка (ср.) | мин–макс | Логит (ср.) | TV (ср.) | Перенос (ср.) | '
              'Одно лицо | Несколько | Узор | Уверенность SCRFD (макс.) | ArcFace-косинус (ср.) |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    def row(name, kind, group):
        score, classes = group['score'], group['classes']
        detected = analysis['detector']['available']
        return (f"| {name} | {kind} | {group['n']} | {_mean(score)} | "
                f"{_fmt(score['min']) if score else 'N/A'}–{_fmt(score['max']) if score else 'N/A'} | "
                f"{_mean(group['logit'], 3)} | {_mean(group['tv'], 3)} | {_mean(group['transfer_score'])} | "
                + (f"{classes['face-like']} | {classes['many faces']} | {classes['pattern']} | "
                   if detected else 'N/A | N/A | N/A | ')
                + f"{_confidence((group['max_confidence'] or {}).get('max'), detected) if group['n'] else 'N/A'} | "
                + f"{_mean(group['arcface_cosine'], 3)} |")
    for name, groups in analysis['per_variant'].items():
        lines.append(row(name, 'финал', groups['final']))
        lines.append(row(name, 'начальное', groups['initial']))
    real = analysis['real']
    if real['available']:
        lines.append(row('реальные test', 'позитивы', real['positive']))
        lines.append(row('реальные test', 'негативы', real['negative']))
    else:
        lines += ['', f"Реальные кропы для сравнения: N/A — {real['reason']}."]
    lines += ['', 'Каждое финальное изображение:', '',
              '| Вариант | Seed | Оценка | Логит | TV | Перенос | Лиц | Уверенность SCRFD | ArcFace | Класс |',
              '|---|---|---|---|---|---|---|---|---|---|']
    for item in (r for r in analysis['images'] if r['kind'] == 'final'):
        lines.append(f"| {item['variant']} | {item['seed']} | {item['score']:.4f} | {item['logit']:.3f} | "
                     f"{item['tv']:.3f} | {_fmt(item['transfer_score'])} | "
                     f"{'N/A' if item['faces'] is None else item['faces']} | "
                     f"{_confidence(item['max_confidence'], item['faces'] is not None)} | "
                     f"{_fmt(item['arcface_cosine'], 3)} | {CLASS_RU[item['class']]} |")
    lines += ['', '### Наблюдения по измеренным значениям', '']
    if summary['technical_test']:
        lines.append('- Технический прогон со случайными весами: значения не описывают обученную модель.')
    lines += ['- '+line for line in summary['interpretation']]
    lines += ['', 'Высокая оценка классификатора сама по себе не означает узнаваемое лицо: это проверяют детектор и ArcFace.', '']
    for seed in summary['seeds']:
        lines += [f'### Начальный шум {seed}', '', f'Строки — варианты, столбцы — шаги оптимизации.', '',
                  f'![Шаги для seed {seed}](synthesis/seed_{seed}/progression.png)', '']
    lines += ['Полные контактные листы и GIF каждого запуска хранятся локально: '
              '`synthesis/<вариант>/seed_<k>/progression.png` и `progression.gif`.', '']
    return lines
