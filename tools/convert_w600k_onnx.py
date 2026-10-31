"""Exact conversion: InsightFace buffalo_l w600k_r50.onnx -> arcface_torch iresnet50 state_dict.

The official ONNX (exported by PyTorch 1.9, opset 11) is the same iresnet50 with every
Conv->BatchNorm pair folded into a biased Conv. The folded kernel goes to the bias-free Conv,
its bias to the following BatchNorm with mean 0, var 1-eps, weight 1 (eval: identity + bias).
Pre-activation bn1, bn2, fc and features are stored unfolded under PyTorch names and copied.
Eval outputs are unchanged. The original statistics of folded pairs are not recoverable, so
train-mode BatchNorm in the backbone does not reproduce the original network (we keep it frozen).
Weights: InsightFace pretrained models, non-commercial research only.

Usage: python tools/convert_w600k_onnx.py [--verify]
"""
from __future__ import annotations
import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

PACK_URL = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
PACK_SHA256 = '80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f'  # 288 621 354 bytes
ONNX_SHA256 = '4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43'
# sokolov.model.state_fingerprint of the converted state_dict (independent of torch.save format).
STATE_SHA256 = '970e13178e114fad152a3fce25c7a464144fc6a7412583126dcd3983dc62e294'
OUTPUT = 'checkpoints/arcface/iresnet50_w600k.pt'
WEIGHTS = 'WebFace600K (InsightFace buffalo_l w600k_r50, converted)'  # model.weights label
EPS = 1e-5
LAYERS = (3, 4, 14, 3)


def onnx_file(root='checkpoints/insightface'):
    """buffalo_l in the insightface layout shared with SCRFD; if absent, downloaded with a socket
    timeout and SHA-256 check (insightface's own downloader has no timeout) and extracted atomically."""
    import shutil
    import zipfile
    from sokolov.io import download
    folder = Path(root) / 'models' / 'buffalo_l'
    if not folder.is_dir():
        archive = download(PACK_URL, folder.with_suffix('.zip'), PACK_SHA256)
        partial = folder.with_name('buffalo_l.part')
        shutil.rmtree(partial, ignore_errors=True)
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(partial)
        partial.replace(folder)
    return folder / 'w600k_r50.onnx'


def convert(path):
    """Map every ONNX initializer onto the iresnet50 state_dict; any mismatch raises ValueError."""
    import numpy as np
    import onnx
    import torch
    from onnx import numpy_helper
    from sokolov.backbones.iresnet import iresnet50
    graph = onnx.load(str(path)).graph
    init = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    users = defaultdict(list)
    for node in graph.node:
        for name in node.input:
            users[name].append(node)
    model = iresnet50(fp16=False)
    modules = dict(model.named_modules())
    state, used, visited = {}, set(), set()

    def take(name):
        used.add(name)
        return init[name]

    def follow(tensor, op):
        nodes = [n for n in users[tensor] if n.op_type == op]
        if len(nodes) != 1:
            raise ValueError(f'{tensor}: expected one {op}, found {len(nodes)}')
        visited.add(id(nodes[0]))
        return nodes[0]

    def attrs(node):
        return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}

    def conv_bn(node, conv, bn):
        layer, a = modules[conv], attrs(node)
        pads = a.get('pads', [0] * 4)
        if (tuple(a['kernel_shape']), tuple(a['strides']), tuple(pads[:2]), a.get('group', 1),
                tuple(a.get('dilations', (1, 1)))) != (layer.kernel_size, layer.stride, layer.padding,
                layer.groups, layer.dilation) or pads[:2] != pads[2:] or len(node.input) != 3:
            raise ValueError(f'{conv}: ONNX Conv geometry differs')
        weight, bias = take(node.input[1]), take(node.input[2])
        size = len(bias)
        state.update({f'{conv}.weight': weight, f'{bn}.weight': np.ones(size, np.float32),
                      f'{bn}.bias': bias, f'{bn}.running_mean': np.zeros(size, np.float32),
                      f'{bn}.running_var': np.full(size, 1 - EPS, np.float32)})
        return node.output[0]

    def batchnorm(node, prefix):
        names = [f'{prefix}.{k}' for k in ('weight', 'bias', 'running_mean', 'running_var')]
        if list(node.input[1:]) != names or abs(attrs(node)['epsilon'] - EPS) > 1e-9:
            raise ValueError(f'{prefix}: unexpected BatchNormalization')
        state.update({name: take(name) for name in names})
        return node.output[0]

    def prelu(node, prefix):
        state[f'{prefix}.weight'] = take(node.input[1]).reshape(-1)
        return node.output[0]

    inputs = [i.name for i in graph.input if i.name not in init]
    if len(inputs) != 1:
        raise ValueError('Expected one image input')
    x = conv_bn(follow(inputs[0], 'Conv'), 'conv1', 'bn1')
    x = prelu(follow(x, 'PRelu'), 'prelu')
    for index, blocks in enumerate(LAYERS, 1):
        for block in range(blocks):
            p = f'layer{index}.{block}'
            y = batchnorm(follow(x, 'BatchNormalization'), f'{p}.bn1')
            y = conv_bn(follow(y, 'Conv'), f'{p}.conv1', f'{p}.bn2')
            y = prelu(follow(y, 'PRelu'), f'{p}.prelu')
            y = conv_bn(follow(y, 'Conv'), f'{p}.conv2', f'{p}.bn3')
            shortcut = (conv_bn(follow(x, 'Conv'), f'{p}.downsample.0', f'{p}.downsample.1')
                        if modules[p].downsample is not None else x)
            add = follow(y, 'Add')
            if sorted(add.input) != sorted([y, shortcut]):
                raise ValueError(f'{p}: residual connection differs')
            x = add.output[0]
    x = batchnorm(follow(x, 'BatchNormalization'), 'bn2')
    flat = follow(x, 'Flatten')
    gemm = follow(flat.output[0], 'Gemm')
    if attrs(flat).get('axis', 1) != 1 or {k: attrs(gemm).get(k, d) for k, d in
            (('alpha', 1.), ('beta', 1.), ('transA', 0), ('transB', 0))} != {'alpha': 1., 'beta': 1., 'transA': 0, 'transB': 1}:
        raise ValueError('fc: unexpected Flatten/Gemm')
    state['fc.weight'], state['fc.bias'] = take(gemm.input[1]), take(gemm.input[2])
    x = batchnorm(follow(gemm.output[0], 'BatchNormalization'), 'features')
    if [o.name for o in graph.output] != [x] or used != set(init) or len(visited) != len(graph.node):
        raise ValueError('ONNX graph has nodes/initializers outside iresnet50')
    result = {}
    for key, value in model.state_dict().items():
        if key.endswith('num_batches_tracked'):
            result[key] = value.clone()
            continue
        array = state.pop(key)
        if tuple(array.shape) != tuple(value.shape) or array.dtype != np.float32:
            raise ValueError(f'{key}: shape {array.shape} != {tuple(value.shape)}')
        result[key] = torch.from_numpy(np.array(array, copy=True))
    if state:
        raise ValueError(f'Unmapped tensors: {sorted(state)[:5]}')
    model.load_state_dict(result, strict=True)
    return result


def convert_checkpoint(output=OUTPUT, root='checkpoints/insightface'):
    """Write the bare state_dict atomically; returns provenance with both SHA-256 values."""
    import torch
    from sokolov.io import sha256, write_json
    from sokolov.model import state_fingerprint
    source = onnx_file(root)
    source_digest = sha256(source)
    if source_digest != ONNX_SHA256:
        raise ValueError(f'Unexpected w600k_r50.onnx SHA-256 {source_digest}')
    state = convert(source)
    fingerprint = state_fingerprint(state)
    if STATE_SHA256 and fingerprint != STATE_SHA256:
        raise ValueError(f'Converted state fingerprint {fingerprint} != {STATE_SHA256}')
    buffer = io.BytesIO()  # fixed in-archive name: bytes do not depend on the target filename
    torch.save(state, buffer)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + '.part')
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(output)
    info = {'source': 'InsightFace buffalo_l/w600k_r50.onnx (ArcFace iResNet50, WebFace600K)',
            'license': 'InsightFace pretrained models: non-commercial research only',
            'onnx': str(source), 'onnx_sha256': source_digest, 'checkpoint': str(output),
            'checkpoint_sha256': sha256(output), 'state_sha256': fingerprint,
            'torch': torch.__version__, 'folded_batchnorm': 'mean=0, var=1-eps, weight=1, bias=folded'}
    write_json(output.with_suffix('.json'), info)
    return info


def sokolov_faces(folder, per_lecture=4):
    """SCRFD + repo align_face + repo ArcFace transform: (uint8 RGB 112x112, normalized tensor)."""
    import numpy as np
    from types import SimpleNamespace
    from PIL import Image
    from sokolov.data.dataset import transforms
    from sokolov.data.prepare import SCRFDDetector, align_face
    detector = SCRFDDetector(SimpleNamespace(detector_name='buffalo_l', detector_threshold=.5,
                                             detector_root=str(ROOT / 'checkpoints/insightface')))
    resize = transforms(SimpleNamespace(model=SimpleNamespace(image_size=112, architecture='arcface')))
    groups = defaultdict(list)
    for path in sorted(Path(folder).glob('*.jpg')):
        groups[path.stem.rsplit('_', 1)[0]].append(path)
    faces = []
    for lecture, paths in sorted(groups.items()):
        found = 0
        for path in paths[::max(1, len(paths) // (4 * per_lecture))]:
            with Image.open(path) as image:
                image = image.convert('RGB')
            detections = detector(image)
            if len(detections) != 1:
                continue
            aligned = align_face(image, detections[0]['landmarks'])
            small = np.asarray(resize.transforms[0](aligned))
            faces.append((lecture, small, resize(aligned).numpy()))
            found += 1
            if found == per_lecture:
                break
    return faces, detector, resize


def verify(output=OUTPUT, root='checkpoints/insightface', faces='data/raw/sokolov_positive_v1/positive'):
    """PyTorch vs onnxruntime on random + real aligned faces; cosine sanity check."""
    import itertools
    import numpy as np
    import onnxruntime
    import torch
    from insightface.model_zoo import get_model
    from sokolov.backbones.iresnet import iresnet50
    torch.set_num_threads(4)
    source = onnx_file(root)
    model = iresnet50(fp16=False)
    model.load_state_dict(torch.load(output, map_location='cpu', weights_only=True), strict=True)
    model.eval()
    session = onnxruntime.InferenceSession(str(source), providers=['CPUExecutionProvider'])
    name = session.get_inputs()[0].name

    def compare(batch):
        with torch.no_grad():
            ours = model(torch.from_numpy(batch)).numpy()
        theirs = np.concatenate([session.run(None, {name: x[None]})[0] for x in batch])
        return ours, float(np.abs(ours - theirs).max()), float(np.abs(theirs).max())

    rng = np.random.default_rng(0)
    noise = rng.uniform(-1, 1, (16, 3, 112, 112)).astype(np.float32)
    _, noise_diff, noise_scale = compare(noise)
    report = {'random_inputs': 16, 'random_max_abs_diff': noise_diff, 'random_max_abs_output': noise_scale}
    found, detector, resize = sokolov_faces(ROOT / faces) if (ROOT / faces).is_dir() else ([], None, None)
    if found:
        lectures = [f[0] for f in found]
        ours, diff, scale = compare(np.stack([f[2] for f in found]))
        insight = get_model(str(source), providers=['CPUExecutionProvider'])
        insight.prepare(ctx_id=-1)
        reference = insight.get_feat([f[1][:, :, ::-1].copy() for f in found])  # insightface takes BGR
        unit = ours / np.linalg.norm(ours, axis=1, keepdims=True)
        same = [float(unit[i] @ unit[j]) for i, j in itertools.combinations(range(len(found)), 2)
                if lectures[i] != lectures[j]]
        report.update({'face_inputs': len(found), 'face_lectures': sorted(set(lectures)),
                       'face_max_abs_diff': diff, 'face_max_abs_output': scale,
                       'insightface_get_feat_max_abs_diff': float(np.abs(ours - reference).max()),
                       'cosine_sokolov_vs_sokolov_cross_lecture': summary(same)})
        others = other_faces(detector, resize)
        if len(others):
            with torch.no_grad():
                other = model(torch.from_numpy(others)).numpy()
            other /= np.linalg.norm(other, axis=1, keepdims=True)
            report.update({'other_faces': len(others), 'other_source': 'insightface package sample images',
                           'cosine_sokolov_vs_other': summary((unit @ other.T).ravel().tolist())})
    return report


def other_faces(detector, resize):
    """Non-Sokolov faces bundled with the insightface package; used for numbers only, never saved."""
    import numpy as np
    import insightface
    from PIL import Image
    from sokolov.data.prepare import align_face
    folder = Path(insightface.__file__).parent / 'data/images'
    tensors = []
    for path in [folder / 't1.jpg', folder / 'Tom_Hanks_54745.png']:
        if path.is_file():
            with Image.open(path) as image:
                image = image.convert('RGB')
            tensors += [resize(align_face(image, d['landmarks'])).numpy() for d in detector(image)]
    return np.stack(tensors) if tensors else np.zeros((0, 3, 112, 112), np.float32)


def summary(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {'pairs': int(values.size), 'mean': float(values.mean()), 'min': float(values.min()),
            'max': float(values.max())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--output', default=str(ROOT / OUTPUT))
    parser.add_argument('--root', default=str(ROOT / 'checkpoints/insightface'))
    parser.add_argument('--verify', action='store_true', help='PyTorch vs onnxruntime and cosine check')
    args = parser.parse_args(argv)
    info = convert_checkpoint(args.output, args.root)
    if args.verify:
        from sokolov.io import write_json
        info['verification'] = verify(args.output, args.root)
        write_json(Path(args.output).with_suffix('.json'), info)
    print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
