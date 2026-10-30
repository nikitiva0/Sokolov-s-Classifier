"""Export both best classifiers + the training SCRFD into an offline Android bundle."""
from pathlib import Path
import shutil
import tempfile
import zipfile

from .io import read_json, sha256, write_json
from .model import build_model
from .synthesize import checkpoint_config, normalized_pixels


def export_mobile(run_dir, output, *, detector=None, allow_technical=False):
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    run, output = Path(run_dir), Path(output)
    if output.exists():
        raise ValueError('Mobile output already exists; choose a new filename')
    checkpoints = []
    for name in ('resnet18', 'arcface'):
        folder = run/name
        if read_json(folder/'status.json')['status'] != 'evaluated':
            raise ValueError(f'Evaluate {name} before mobile export')
        checkpoint = folder/'best.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['technical_test'] and not allow_technical:
            raise ValueError('Refusing a technical/random-weight model; train real classifiers first')
        threshold = read_json(folder/'threshold.json')
        if threshold['checkpoint_sha256'] != sha256(checkpoint) or threshold['selection_partition'] != 'val':
            raise ValueError('Mobile threshold is not bound to the best validation checkpoint')
        checkpoints.append((name, checkpoint_config(saved), threshold, saved))
    if checkpoints[0][3]['data_fingerprint'] != checkpoints[1][3]['data_fingerprint']:
        raise ValueError('Mobile models must share the same dataset/split')
    if checkpoints[0][3]['technical_test'] != checkpoints[1][3]['technical_test']:
        raise ValueError('Cannot mix real and technical models')
    data = checkpoints[0][1].data
    detector = Path(detector) if detector else Path(data.detector_root)/'models'/data.detector_name/'det_10g.onnx'
    if not detector.is_file():
        raise FileNotFoundError('Training SCRFD det_10g.onnx required; pass --detector')
    expected = read_json(run/'data/data_audit.json')['detector']
    if sha256(detector) != expected['sha256']:
        raise ValueError('Detector differs from the one used to prepare training data')
    detector_session = ort.InferenceSession(str(detector), providers=['CPUExecutionProvider'])
    if len(detector_session.get_inputs()) != 1 or len(detector_session.get_outputs()) != 9:
        raise ValueError('Expected SCRFD det_10g with 9 score/box/landmark outputs')
    # Verify the actual anchor layout consumed by the Android decoder.
    detector_input_name = detector_session.get_inputs()[0].name
    detector_outputs = detector_session.run(None, {detector_session.get_inputs()[0].name:
                                                  np.zeros((1, 3, 640, 640), dtype=np.float32)})
    for i, stride in enumerate((8, 16, 32)):
        count = (640//stride)**2*2
        if any(detector_outputs[i+j*3].size != count*width for j, width in enumerate((1, 4, 10))):
            raise ValueError('SCRFD output layout incompatible with Android decoder')
    del detector_session
    manifest = {'schema_version': 1, 'technical_test': checkpoints[0][3]['technical_test'],
        'default_model': 'resnet18', 'class_names': ['not_sokolov', 'sokolov'],
        'preprocessing': 'RGB NCHW float32 [0,1]; model-specific normalization is inside ONNX',
        'alignment': 'five-point ArcFace similarity transform to 224x224; resize to classifier input',
        'models': [], 'detector': {'file': 'detector.onnx', 'sha256': sha256(detector),
            'input_size': 640, 'threshold': data.detector_threshold, 'nms_threshold': .4,
            'input_name': detector_input_name, 'strides': [8, 16, 32], 'anchors': 2},
        'note': 'Thresholds selected on validation. Mobile latency must be measured on the target phone.'}
    with tempfile.TemporaryDirectory(prefix='sokolov-mobile-') as temporary:
        folder = Path(temporary)
        shutil.copyfile(detector, folder/'detector.onnx')
        for name, config, threshold, saved in checkpoints:
            model = build_model(config.model, pretrained=False).cpu().eval()
            model.load_state_dict(saved['model'], strict=True)
            class MobileClassifier(torch.nn.Module):
                def __init__(self, network, architecture):
                    super().__init__(); self.network = network; self.architecture = architecture
                def forward(self, x):
                    return self.network(normalized_pixels(x, self.architecture)).sigmoid()
            wrapper = MobileClassifier(model, name).eval()
            size = config.model.image_size
            sample = torch.rand((1, 3, size, size), generator=torch.Generator().manual_seed(42))
            target = folder/f'{name}.onnx'
            with torch.no_grad():
                torch.onnx.export(wrapper, sample, target, input_names=['pixels'], output_names=['score'],
                                  opset_version=17, do_constant_folding=True, dynamo=False)
            onnx.checker.check_model(str(target))
            session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
            parity = []
            for pixels in (sample, torch.zeros_like(sample), torch.ones_like(sample)):
                with torch.no_grad(): expected_score = wrapper(pixels).numpy()
                actual = session.run(['score'], {'pixels': pixels.numpy()})[0]
                difference = float(np.max(np.abs(expected_score-actual)))
                if not np.isfinite(actual).all() or difference > 1e-4:
                    raise ValueError(f'{name}: ONNX/PyTorch predictions differ by {difference}')
                parity.append(difference)
            manifest['models'].append({'id': name, 'file': target.name, 'sha256': sha256(target),
                'size_bytes': target.stat().st_size, 'input_size': size, 'input_name': 'pixels',
                'output_name': 'score', 'threshold': threshold['threshold'],
                'source_checkpoint_sha256': sha256(run/name/'best.pt'),
                'validation_ap': saved['best_ap'], 'epoch': saved['epoch']+1,
                'max_parity_error': max(parity)})
            del session, model, wrapper
        write_json(folder/'manifest.json', manifest)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_zip = output.with_suffix('.partial.zip')
        with zipfile.ZipFile(temporary_zip, 'w', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(folder.iterdir()): archive.write(path, path.name)
        temporary_zip.replace(output)
    write_json(output.with_suffix('.json'), manifest)
    return output
