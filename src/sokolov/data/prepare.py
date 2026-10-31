"""Audit, group split, one shared SCRFD/5-point alignment and a near-duplicate guard."""
from __future__ import annotations
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from ..contracts import RawSample
from ..io import read_csv, read_json, sha256, write_csv, write_json
from .dedup import close_pairs, nearest, phash64
from .download import image_info, legacy_fields
from .families import source_family
from .split import SPLITS, split_groups, split_summary

# 3: merged prepared manifest (primary face, crop pHash, near_duplicate status, pHash-linked groups).
SCHEMA_VERSION = 3
DETECTOR_INPUT_SIZE = 640
DETECTOR_RESIZE_POLICY = 'shrink_only_top_left_pad_v1'
LEGACY_WARNINGS = [
    'Random frame split may share lectures and near-duplicate frames across train/val/test.',
    'FairFace identity-disjoint splits are not guaranteed.',
    'Positive and negative classes come from different sources; domain shortcuts are possible.',
    'Test is not an independent new-lecture benchmark.',
]
GROUP_WARNINGS = [
    'FairFace не размечен по личностям: один человек может оказаться в разных файлах и разных выборках.',
    'Посторонние люди в кадрах источника Соколова сгруппированы по источнику; один и тот же неизвестный '
    'человек из разных видео может попасть в разные выборки.',
    'Независимых источников Соколова немного: в test лишь несколько групп-позитивов, доверительные интервалы широкие.',
    'Источники, связанные pHash-кандидатами исходных изображений, объединяются в одну группу; pHash-фильтр '
    'выровненных лиц исключает из оценки только почти идентичные кропы. Похожие кадры одного человека из разных '
    'видео остаются отдельными группами.',
    'Личность на позитивах подтверждена визуально и сходством ArcFace-эмбеддингов, а не независимой разметкой.',
]
TRAIN_ONLY_WARNING = ('Группы split.train_only_sources и закреплённые за train используются только для обучения; '
                      'это известные примеры разработки, а не независимая проверка.')
RAW_CANDIDATES_KEPT = 1000


def detector_input(image, size=DETECTOR_INPUT_SIZE):
    """Keep small faces at their native scale; shrink large images and pad black.

    Return BGR input and the exact height scale used by InsightFace's decoder.
    Enlarging a tight 224px FairFace crop to 640px makes its face too large for
    this detector. This policy depends on image geometry, never on its label.
    """
    import cv2
    import numpy as np
    bgr = np.asarray(image)[:, :, ::-1].copy()
    height, width = bgr.shape[:2]
    scale = min(1., size/width, size/height)
    new_width, new_height = max(1, int(width*scale)), max(1, int(height*scale))
    resized = cv2.resize(bgr, (new_width, new_height))
    padded = np.zeros((size, size, 3), dtype=np.uint8)
    padded[:new_height, :new_width] = resized
    return padded, new_height/height


def validate_detector_coverage(audit, minimum):
    """Fail before training when either class is mostly lost to preprocessing.

    Faces excluded later as cross-split near-duplicates were detected and aligned;
    they are removed from the denominator instead of counting as detector losses.
    """
    failures = []
    for split, counts in audit['counts'].items():
        for label, name in [('0', 'not_sokolov'), ('1', 'sokolov')]:
            total = (counts['labels_original'].get(label, 0) -
                     counts.get('labels_near_duplicate', {}).get(label, 0))
            usable = counts['labels_usable'].get(label, 0)
            if not total or usable/total < minimum:
                failures.append(f'{split}/{name}: {usable}/{total}')
    if failures:
        raise ValueError(f'Detector coverage below {minimum:.0%}: ' + ', '.join(failures)
                         + '. See data_audit.json; fix preprocessing before training.')


class SCRFDDetector:
    def __init__(self, data):
        import onnxruntime as ort
        from insightface.model_zoo.scrfd import SCRFD
        from insightface.utils.storage import ensure_available
        root = Path(ensure_available('models', data.detector_name, root=data.detector_root))
        path = root / 'det_10g.onnx'
        if not path.is_file():
            raise FileNotFoundError('buffalo_l SCRFD det_10g.onnx missing')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.model = SCRFD(str(path), session=ort.InferenceSession(
            str(path), sess_options=options, providers=['CPUExecutionProvider']))
        self.model.prepare(ctx_id=-1, input_size=(DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE),
                           det_thresh=data.detector_threshold)
        self.metadata = {'name': 'SCRFD/det_10g.onnx', 'insightface': version('insightface'),
                         'sha256': sha256(path), 'threshold': data.detector_threshold,
                         'input_size': [DETECTOR_INPUT_SIZE, DETECTOR_INPUT_SIZE],
                         'resize_policy': DETECTOR_RESIZE_POLICY, 'provider': 'CPUExecutionProvider',
                         'intra_op_num_threads': 4}

    def __call__(self, image):
        padded, scale = detector_input(image)
        boxes, landmarks = self.model.detect(padded, max_num=0)
        if len(boxes) == 0:
            return []
        if landmarks is None:
            raise RuntimeError('SCRFD model did not return five-point landmarks')
        return [{'bbox': (box[:4]/scale).tolist(), 'confidence': float(box[4]),
                 'landmarks': (points/scale).tolist()} for box, points in zip(boxes, landmarks)]


def align_face(image, landmarks):
    """ArcFace canonical five landmarks; RGB affine warp to 224x224."""
    import cv2
    import numpy as np
    from skimage.transform import SimilarityTransform
    from PIL import Image
    target = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                       [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float64) * 2
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        raise ValueError('Expected five finite landmarks')
    transform = SimilarityTransform()
    if not transform.estimate(points, target) or not np.isfinite(transform.params).all():
        raise ValueError('Cannot align face')
    return Image.fromarray(cv2.warpAffine(np.asarray(image), transform.params[:2], (224, 224),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0))


def primary_face(faces, size):
    """Index of the target face: bbox centre nearest the image centre; tie -> larger box."""
    width, height = size
    def key(item):
        index, face = item
        x1, y1, x2, y2 = face['bbox']
        distance = math.hypot((x1+x2)/2 - width/2, (y1+y2)/2 - height/2)
        return (round(distance, 6), -(x2-x1)*(y2-y1), -face['confidence'], index)
    return min(enumerate(faces), key=key)[0]


def near_duplicate_guard(rows, max_distance):
    """Mark val/test crops within max_distance bits of train (and test of val) as near_duplicate.

    Mutates rows; O(N_query * N_reference) Hamming comparisons in bounded chunks.
    """
    ok = {split: [r for r in rows if r['split'] == split and r['status'] == 'ok'] for split in SPLITS}
    hashes = {split: [int(r['crop_phash'], 16) for r in ok[split]] for split in SPLITS}
    pairs = (('val', 'train'), ('test', 'train'), ('test', 'val'))
    before, flagged = {}, []
    for query, reference in pairs:
        distance, index = nearest(hashes[query], hashes[reference])
        key = f'{query}-{reference}'
        before[key] = int(distance.min()) if len(hashes[query]) and len(hashes[reference]) else None
        for row, d, i in zip(ok[query], distance, index):
            if d <= max_distance and row['status'] == 'ok':
                row.update(status='near_duplicate', near_duplicate_of=ok[reference][i]['image_id'],
                           near_duplicate_distance=int(d))
                flagged.append({'image_id': row['image_id'], 'split': query, 'label': row['label'],
                                'domain': row['domain'], 'nearest': ok[reference][i]['image_id'],
                                'nearest_split': reference, 'distance': int(d)})
    remaining = {split: [int(r['crop_phash'], 16) for r in rows if r['split'] == split and r['status'] == 'ok']
                 for split in SPLITS}
    after = {}
    for query, reference in pairs:
        distance, _ = nearest(remaining[query], remaining[reference])
        after[f'{query}-{reference}'] = int(distance.min()) if len(remaining[query]) and len(remaining[reference]) else None
    counts = Counter((f['split'], str(f['label'])) for f in flagged)
    return {'hash': 'pHash 64 bit (imagehash, hash_size=8) of aligned 224x224 crop', 'max_distance': max_distance,
            'rule': 'val vs train, test vs train, test vs val (all ok val crops)',
            'excluded': {split: {label: counts[(split, label)] for label in ('0', '1')} for split in ('val', 'test')},
            'excluded_total': len(flagged), 'min_cross_split_distance_before': before,
            'min_cross_split_distance_after': after}, flagged


def link_groups(samples, groups, pairs):
    """Union-find over pHash-linked raw images of different groups; root = smallest group id.

    O((N + P) α(N)) for N groups and P linked pairs.
    """
    parents = {group: group for group in groups.values()}
    def root(group):
        while parents[group] != group:
            parents[group] = parents[parents[group]]
            group = parents[group]
        return group
    for i, j, _ in pairs:
        left, right = root(groups[samples[i].image_id]), root(groups[samples[j].image_id])
        if left != right:
            parents[max(left, right)] = min(left, right)
    merged = {image_id: root(group) for image_id, group in groups.items()}
    members = defaultdict(set)
    for group in parents:
        members[root(group)].add(group)
    return merged, {r: sorted(m) for r, m in sorted(members.items()) if len(m) > 1}


def _warnings(samples, rows, mode, pinned):
    if any(s.partition for s in samples):
        return LEGACY_WARNINGS[1:3]
    if mode != 'group_stratified':
        return LEGACY_WARNINGS
    labels = defaultdict(set)
    for row in rows:
        if row['status'] == 'ok':
            labels[row['domain']].add(row['label'])
    single = [f"{domain or 'unknown'} ({next(iter(v))})" for domain, v in sorted(labels.items()) if len(v) == 1]
    extra = [f"Домены только с одним классом: {', '.join(single)}. Модель может опираться на признаки домена; "
             'см. разбивку по доменам в отчёте.'] if single else []
    return GROUP_WARNINGS + ([TRAIN_ONLY_WARNING] if pinned else []) + extra


def _sample(row, path):
    dataset, domain = legacy_fields(row)
    return RawSample(row['image_id'], path, int(row['label']), row['source_id'], row.get('source_url', ''),
                     row.get('partition') or None, row.get('group_id', ''), row.get('domain') or domain,
                     row.get('dataset') or dataset, row.get('person', ''), row.get('pinned_split', ''))


def _cached_detections(config, detector):
    """image_id -> row of a verified earlier prepared manifest with the same detector (data.reuse_prepared_manifest)."""
    if not config.data.reuse_prepared_manifest:
        return {}
    cached_path = Path(config.data.reuse_prepared_manifest)
    cached_metadata = read_json(cached_path.parent/'prepared.json')
    if cached_metadata['manifest_sha256'] != sha256(cached_path):
        raise ValueError('Reuse manifest was modified')
    if cached_metadata['detector'] != detector.metadata:
        raise ValueError('Cannot reuse crops from a different detector/policy')
    return {r['image_id']: r for r in read_csv(cached_path)}


def prepare(config, assignments=None, *, detector=None):
    dest = Path(config.data.prepared_manifest)
    folder = dest.parent
    raw_path = Path(config.data.raw_manifest)
    signature = {'raw_sha256': sha256(raw_path), 'split': asdict(config.split),
                 'detector_name': config.data.detector_name,
                 'detector_threshold': config.data.detector_threshold,
                 'phash_max_distance': config.data.phash_max_distance,
                 'detector_resize_policy': DETECTOR_RESIZE_POLICY,
                 'min_detector_coverage': config.data.min_detector_coverage, 'schema_version': SCHEMA_VERSION}
    completion = folder / 'prepared.json'
    if completion.exists():
        saved = read_json(completion)
        if saved['signature'] != signature:
            raise ValueError('Preparation config/input changed; choose a new prepared_manifest directory')
        if saved['manifest_sha256'] != sha256(dest):
            raise ValueError('Prepared manifest modified')
        for row in read_csv(dest):
            if image_info(row['path'])[1] != row['pixel_sha256']:
                raise ValueError('Original image modified after preparation')
            if row['crop_path'] and (not Path(row['crop_path']).is_file() or
                    sha256(row['crop_path']) != row['crop_sha256']):
                raise ValueError('Prepared crop missing or modified; choose a new preparation directory')
        validate_detector_coverage(read_json(folder / 'data_audit.json'), config.data.min_detector_coverage)
        return dest
    samples, seen, phashes, exclusions, digests = [], {}, [], [], {}
    raw_rows = read_csv(raw_path)
    write_json(folder / 'prepare_progress.json', {
        'phase': 'audit', 'processed': 0, 'total': len(raw_rows), 'completed': False})
    ids = set()
    for row in raw_rows:
        image_id = row['image_id']
        if image_id in ids:
            raise ValueError(f'Duplicate image_id: {image_id}')
        ids.add(image_id)
        if row['label'] not in ('0', '1') or not row['source_id']:
            raise ValueError(f'Invalid label/source_id: {image_id}')
        if row.get('partition', '') not in ('', 'train', 'test'):
            raise ValueError(f'Invalid partition: {image_id}')
        path = Path(row['path']).resolve()
        try:
            image, digest = image_info(path)
        except (OSError, ValueError):
            exclusions.append({'image_id': image_id, 'reason': 'unreadable'})
            continue
        if row.get('pixel_sha256') and row['pixel_sha256'] != digest:
            raise ValueError(f'Image changed after selection: {image_id}')
        if digest in seen:
            previous = seen[digest]
            if previous['label'] != row['label']:
                raise ValueError(f'Conflicting duplicate labels: {image_id}')
            if previous.get('partition', '') != row.get('partition', ''):
                raise ValueError('Duplicate crosses external partitions')
            exclusions.append({'image_id': image_id, 'reason': 'exact_duplicate',
                               'duplicate_of': previous['image_id']})
            continue
        seen[digest] = row
        digests[image_id] = digest
        phashes.append(phash64(image))
        samples.append(_sample(row, path))
    groups = {s.image_id: s.group for s in samples}
    # Raw-level pHash candidates: only cross-group pairs matter once groups stay together.
    pairs, within = [], 0
    for i, j, distance in close_pairs(phashes, config.data.phash_max_distance):
        if groups[samples[i].image_id] == groups[samples[j].image_id]:
            within += 1
        else:
            pairs.append((i, j, distance))
    linked = {}
    if config.split.mode == 'group_stratified':
        # Linked sources share one split: groups connected by a candidate pair are merged.
        groups, linked = link_groups(samples, groups, pairs)
    near = [{'left': samples[i].image_id, 'right': samples[j].image_id, 'distance': d} for i, j, d in pairs]
    near_total = len(near)
    assigned = assignments or split_groups(samples, groups, config.split)
    mapping = {a.image_id: a for a in assigned}
    if set(mapping) != {s.image_id for s in samples}:
        raise ValueError('Assignments do not match audited source images')
    near_cross_split = [p for p in near if mapping[p['left']].split != mapping[p['right']].split]
    near = sorted(near, key=lambda item: (item['distance'], item['left'], item['right']))[:RAW_CANDIDATES_KEPT]
    split_path = folder / 'split.csv'
    if split_path.exists():
        expected = [{k: str(v) if v is not None else '' for k, v in asdict(a).items()} for a in assigned]
        if read_csv(split_path) != expected:
            raise ValueError('Existing split differs; use a new preparation directory')
    else:
        write_csv(split_path, [asdict(a) for a in assigned])
    summary = split_summary(samples, assigned, config.split)
    write_json(folder / 'split_summary.json', summary)
    detector = detector or SCRFDDetector(config.data)
    cached_rows = _cached_detections(config, detector)
    rows = []
    start = time.perf_counter()
    write_json(folder / 'prepare_progress.json', {
        'phase': 'detection', 'processed': 0, 'total': len(samples), 'completed': False})
    for i, sample in enumerate(samples):
        assignment = mapping[sample.image_id]
        # IDs may originate in an external manifest: never use them as filesystem paths.
        filename = hashlib.sha256(sample.image_id.encode()).hexdigest() + '.png'
        crop = folder / 'faces' / filename
        digest = digests[sample.image_id]
        cached = cached_rows.get(sample.image_id)
        reused = bool(cached and cached['pixel_sha256'] == digest and int(cached['label']) == sample.label)
        image = None
        if reused:
            faces = json.loads(cached['face_metadata'])
            detection_seconds = 0.
        else:
            image, current_digest = image_info(sample.path)
            if current_digest != digest:
                raise ValueError('Image changed during preparation')
            before = time.perf_counter()
            faces = detector(image)
            detection_seconds = time.perf_counter() - before
        status, primary, crop_phash, crop_sha = 'no_face', '', '', ''
        if faces:
            if image is None and len(faces) > 1:
                image = image_info(sample.path)[0]
            primary = primary_face(faces, image.size) if len(faces) > 1 else 0
            previous = (cached.get('crop_path') if reused and str(cached.get('primary_face', '0') or '0') == str(primary)
                        else '')
            if previous:
                # Same pixels, detector and target face: the earlier aligned crop is identical.
                crop = Path(previous)
                crop_sha = sha256(crop)
                if crop_sha != cached['crop_sha256']:
                    raise ValueError('Reuse crop was modified')
                from PIL import Image
                with Image.open(crop) as aligned:
                    crop_phash = cached.get('crop_phash') or f'{phash64(aligned.convert("RGB")):016x}'
                status = 'ok'
            else:
                if image is None:
                    image = image_info(sample.path)[0]
                try:
                    aligned = align_face(image, faces[primary]['landmarks'])
                except ValueError:
                    status = 'alignment_failed'
                else:
                    status = 'ok'
                    crop.parent.mkdir(parents=True, exist_ok=True)
                    aligned.save(crop)
                    crop_sha = sha256(crop)
                    crop_phash = f'{phash64(aligned):016x}'
        rows.append({'image_id': sample.image_id, 'path': str(sample.path), 'label': sample.label,
            'source_id': sample.source_id, 'group_id': assignment.group_id, 'domain': sample.domain,
            'dataset': sample.dataset, 'person': sample.person, 'source_url': sample.source_url,
            'partition': sample.partition or '', 'split': assignment.split, 'variant': 'face_crop',
            'status': status, 'pixel_sha256': digest, 'crop_path': str(crop.resolve()) if status == 'ok' else '',
            'crop_sha256': crop_sha if status == 'ok' else '', 'crop_phash': crop_phash,
            'face_count': len(faces), 'primary_face': primary, 'face_metadata': json.dumps(faces),
            'detection_seconds': detection_seconds, 'detection_reused': reused,
            'near_duplicate_of': '', 'near_duplicate_distance': ''})
        if (i+1) % 25 == 0 or i+1 == len(samples):
            write_json(folder / 'prepare_progress.json', {
                'phase': 'detection', 'processed': i+1, 'total': len(samples),
                'completed': False, 'statuses': dict(Counter(r['status'] for r in rows))})
        if (i+1) % 100 == 0:
            print(f'Prepared {i+1}/{len(samples)}', flush=True)
    guard, flagged = near_duplicate_guard(rows, config.split.dedup_max_distance)
    stats = {}
    for split in SPLITS:
        subset = [r for r in rows if r['split'] == split]
        usable = [r for r in subset if r['status'] == 'ok']
        stats[split] = {'originals': len(subset), 'usable': len(usable),
            'labels_original': dict(Counter(str(r['label']) for r in subset)),
            'labels_usable': dict(Counter(str(r['label']) for r in usable)),
            'labels_near_duplicate': dict(Counter(str(r['label']) for r in subset if r['status'] == 'near_duplicate')),
            'statuses': dict(Counter(r['status'] for r in subset)),
            'several_faces_primary_used': sum(r['status'] == 'ok' and r['face_count'] > 1 for r in subset)}
        stats[split]['coverage_by_label'] = {label:
            stats[split]['labels_usable'].get(label, 0) /
            max(1, total - stats[split]['labels_near_duplicate'].get(label, 0))
            for label, total in stats[split]['labels_original'].items()}
        web = [r for r in subset if source_family(r) == 'web']
        stats[split]['web_by_label'] = dict(Counter(str(r['label']) for r in web))
        stats[split]['web_groups_by_label'] = {str(label): len({r['group_id'] for r in web if r['label'] == label})
                                               for label in (0, 1)}
    faces = Counter((r['split'], r['label'], r['domain'], r['status']) for r in rows)
    for cell in summary['cells']:
        key = (cell['split'], cell['label'], cell['domain'])
        cell['faces_ok'] = faces[key + ('ok',)]
        cell['near_duplicate'] = faces[key + ('near_duplicate',)]
    summary['near_duplicate_guard'] = guard
    summary['phash_linked_groups'] = len(linked)
    write_json(folder / 'split_summary.json', summary)
    source_splits, group_splits = defaultdict(set), defaultdict(set)
    for row in rows:
        source_splits[row['source_id']].add(row['split'])
        group_splits[row['group_id']].add(row['split'])
    pinned = bool(config.split.train_only_sources) or any(s.pinned_split for s in samples)
    audit = {'input_count': len(raw_rows), 'audited_count': len(samples), 'counts': stats,
        'split_mode': config.split.mode, 'split_summary': summary,
        'exclusions': exclusions, 'near_duplicate_candidates': near,
        'near_duplicate_candidate_counts': {'max_distance': config.data.phash_max_distance,
            'within_group_pairs': within, 'cross_group_pairs': near_total, 'cross_group_kept': len(near)},
        'near_duplicate_cross_split': near_cross_split,
        'phash_linked_groups': dict(list(linked.items())[:RAW_CANDIDATES_KEPT]),
        'near_duplicate_guard': guard, 'near_duplicate_excluded': flagged,
        'overlapping_sources': {k: sorted(v) for k, v in source_splits.items() if len(v) > 1},
        'overlapping_groups': {k: sorted(v) for k, v in group_splits.items() if len(v) > 1},
        'reused_detections': sum(r['detection_reused'] for r in rows),
        'warnings': _warnings(samples, rows, config.split.mode, pinned),
        'detector': detector.metadata, 'primary_face_rule': 'nearest bbox centre to image centre; tie -> larger box',
        'detection_seconds': sum(r['detection_seconds'] for r in rows),
        'preparation_seconds': time.perf_counter()-start}
    write_json(folder / 'data_audit.json', audit)
    write_csv(dest, rows)
    for split in SPLITS:
        if set(stats[split]['labels_usable']) != {'0', '1'}:
            raise ValueError(f'Need both usable face classes in {split}; see data_audit.json')
    validate_detector_coverage(audit, config.data.min_detector_coverage)
    write_json(completion, {'signature': signature, 'manifest_sha256': sha256(dest),
                            'split_sha256': sha256(split_path), 'detector': detector.metadata})
    write_json(folder / 'prepare_progress.json', {
        'phase': 'completed', 'processed': len(samples), 'total': len(samples), 'completed': True})
    return dest
