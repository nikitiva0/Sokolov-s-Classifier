"""Audit, split, and one shared SCRFD/5-point alignment for both classes."""
from __future__ import annotations
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from ..contracts import RawSample
from ..io import read_csv, read_json, sha256, write_csv, write_json
from .download import image_info
from .split import split_groups

WARNINGS = [
    'Random frame split may share lectures and near-duplicate frames across train/val/test.',
    'FairFace identity-disjoint splits are not guaranteed.',
    'Positive and negative classes come from different sources; domain shortcuts are possible.',
    'Test is not an independent new-lecture benchmark.',
]

DETECTOR_INPUT_SIZE = 640
DETECTOR_RESIZE_POLICY = 'shrink_only_top_left_pad_v1'


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
    """Fail before training when either class is mostly lost to preprocessing."""
    failures = []
    for split, counts in audit['counts'].items():
        for label, name in [('0', 'not_sokolov'), ('1', 'sokolov')]:
            total = counts['labels_original'].get(label, 0)
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


def prepare(config, assignments=None, *, detector=None):
    import imagehash
    dest = Path(config.data.prepared_manifest)
    folder = dest.parent
    raw_path = Path(config.data.raw_manifest)
    signature = {'raw_sha256': sha256(raw_path), 'split': asdict(config.split),
                 'detector_name': config.data.detector_name,
                 'detector_threshold': config.data.detector_threshold,
                 'phash_max_distance': config.data.phash_max_distance,
                 'detector_resize_policy': DETECTOR_RESIZE_POLICY,
                 'min_detector_coverage': config.data.min_detector_coverage, 'schema_version': 2}
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
            if row['status'] == 'ok' and (not Path(row['crop_path']).is_file() or
                    sha256(row['crop_path']) != row['crop_sha256']):
                raise ValueError('Prepared crop missing or modified; choose a new preparation directory')
        validate_detector_coverage(read_json(folder / 'data_audit.json'), config.data.min_detector_coverage)
        return dest
    samples, seen, phashes, exclusions = [], {}, [], []
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
        phashes.append((image_id, int(str(imagehash.phash(image)), 16)))
        samples.append(RawSample(image_id, path, int(row['label']), row['source_id'],
                                 row.get('source_url', ''), row.get('partition') or None))
    near = []
    for i, (image_id, value) in enumerate(phashes):
        for other_id, other_value in phashes[:i]:
            distance = (value ^ other_value).bit_count()
            if distance <= config.data.phash_max_distance:
                near.append({'left': other_id, 'right': image_id, 'distance': distance})
    groups = {s.image_id: s.source_id for s in samples}
    assigned = assignments or split_groups(samples, groups, config.split)
    mapping = {a.image_id: a for a in assigned}
    if set(mapping) != {s.image_id for s in samples}:
        raise ValueError('Assignments do not match audited source images')
    split_path = folder / 'split.csv'
    if split_path.exists():
        expected = [{k: str(v) if v is not None else '' for k, v in asdict(a).items()} for a in assigned]
        if read_csv(split_path) != expected:
            raise ValueError('Existing split differs; use a new preparation directory')
    else:
        write_csv(split_path, [asdict(a) for a in assigned])
    detector = detector or SCRFDDetector(config.data)
    rows = []
    start = time.perf_counter()
    write_json(folder / 'prepare_progress.json', {
        'phase': 'detection', 'processed': 0, 'total': len(samples), 'completed': False})
    for i, sample in enumerate(samples):
        assignment = mapping[sample.image_id]
        # IDs may originate in an external manifest: never use them as filesystem paths.
        filename = hashlib.sha256(sample.image_id.encode()).hexdigest() + '.png'
        crop = folder / 'faces' / filename
        image, digest = image_info(sample.path)
        before = time.perf_counter()
        faces = detector(image)
        detection_seconds = time.perf_counter() - before
        status = 'ok' if len(faces) == 1 else ('no_face' if not faces else 'multiple_faces')
        face = faces[0] if status == 'ok' else None
        if face:
            try:
                aligned = align_face(image, face['landmarks'])
            except ValueError:
                status = 'alignment_failed'
            else:
                crop.parent.mkdir(parents=True, exist_ok=True)
                aligned.save(crop)
        import json
        rows.append({'image_id': sample.image_id, 'path': str(sample.path), 'label': sample.label,
            'source_id': sample.source_id, 'source_url': sample.source_url,
            'partition': sample.partition or '', 'split': assignment.split,
            'group_id': assignment.group_id, 'variant': 'face_crop', 'status': status,
            'pixel_sha256': digest, 'crop_path': str(crop.resolve()) if status == 'ok' else '',
            'crop_sha256': sha256(crop) if status == 'ok' else '', 'face_count': len(faces),
            'face_metadata': json.dumps(faces), 'detection_seconds': detection_seconds})
        if (i+1) % 25 == 0 or i+1 == len(samples):
            write_json(folder / 'prepare_progress.json', {
                'phase': 'detection', 'processed': i+1, 'total': len(samples),
                'completed': False, 'statuses': dict(Counter(r['status'] for r in rows))})
        if (i+1) % 100 == 0:
            print(f'Prepared {i+1}/{len(samples)}', flush=True)
    stats = {}
    for split in ('train', 'val', 'test'):
        subset = [r for r in rows if r['split'] == split]
        usable = [r for r in subset if r['status'] == 'ok']
        stats[split] = {'originals': len(subset), 'usable': len(usable),
            'labels_original': dict(Counter(str(r['label']) for r in subset)),
            'labels_usable': dict(Counter(str(r['label']) for r in usable)),
            'statuses': dict(Counter(r['status'] for r in subset))}
        stats[split]['coverage_by_label'] = {label:
            stats[split]['labels_usable'].get(label, 0)/total
            for label, total in stats[split]['labels_original'].items()}
    source_splits = defaultdict(set)
    for row in rows:
        source_splits[row['source_id']].add(row['split'])
    audit = {'input_count': len(raw_rows), 'audited_count': len(samples), 'counts': stats,
        'exclusions': exclusions, 'near_duplicate_candidates': near,
        'overlapping_sources': {k: sorted(v) for k, v in source_splits.items() if len(v) > 1},
        'warnings': WARNINGS if not any(s.partition for s in samples) else WARNINGS[1:3],
        'detector': detector.metadata, 'detection_seconds': sum(r['detection_seconds'] for r in rows),
        'preparation_seconds': time.perf_counter()-start}
    write_json(folder / 'data_audit.json', audit)
    write_csv(dest, rows)
    for split in ('train', 'val', 'test'):
        if set(stats[split]['labels_usable']) != {'0', '1'}:
            raise ValueError(f'Need both usable face classes in {split}; see data_audit.json')
    validate_detector_coverage(audit, config.data.min_detector_coverage)
    write_json(completion, {'signature': signature, 'manifest_sha256': sha256(dest),
                            'split_sha256': sha256(split_path), 'detector': detector.metadata})
    write_json(folder / 'prepare_progress.json', {
        'phase': 'completed', 'processed': len(samples), 'total': len(samples), 'completed': True})
    return dest
