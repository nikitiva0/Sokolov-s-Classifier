"""PyTorch ResNet18 and official ArcFace iResNet50, each producing one logit."""
from __future__ import annotations
import hashlib
from pathlib import Path

from .io import download, sha256


def state_fingerprint(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def ensure_arcface_checkpoint(config):
    path = Path(config.checkpoint)
    if not path.exists() and config.checkpoint_url:
        download(config.checkpoint_url, path, config.checkpoint_sha256)
    if not path.is_file():
        raise FileNotFoundError('Attach official iResNet50/MS1MV3 PyTorch checkpoint and set model.checkpoint; '
                                'see docs/KAGGLE.md. No random-weight fallback is allowed.')
    digest = sha256(path)
    if config.checkpoint_sha256 and digest != config.checkpoint_sha256:
        raise ValueError('ArcFace checkpoint SHA-256 mismatch')
    return path, digest


def build_model(config, *, pretrained=True):
    import torch
    from torch import nn
    if config.architecture == 'resnet18':
        from torchvision.models import resnet18, ResNet18_Weights
        model = resnet18(weights=None)
        if pretrained:
            weights = ResNet18_Weights.IMAGENET1K_V1
            state = weights.get_state_dict(progress=True, check_hash=True)
            model.load_state_dict(state, strict=True)
            provenance = {'source': weights.url, 'state_sha256': state_fingerprint(state)}
        model.fc = nn.Linear(512, 1)
    elif config.architecture == 'arcface':
        from .backbones.iresnet import iresnet50
        backbone = iresnet50(fp16=False)
        if not pretrained:
            # Stable random initialization for offline technical tests; saved weights
            # completely replace this state when restoring a trained checkpoint.
            for layer in backbone.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
        if pretrained:
            path, digest = ensure_arcface_checkpoint(config)
            state = torch.load(path, map_location='cpu', weights_only=True)
            # Official files are bare state_dicts; allow a container with explicit state_dict.
            if 'state_dict' in state:
                state = state['state_dict']
            if all(k.startswith('module.') for k in state):
                state = {k.removeprefix('module.'): v for k, v in state.items()}
            backbone.load_state_dict(state, strict=True)
            provenance = {'source': 'InsightFace arcface_torch iResNet50/MS1MV3', 'sha256': digest}

        class FaceClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                self.head = nn.Linear(512, 1)

            def forward(self, x):
                return self.head(torch.nn.functional.normalize(self.backbone(x), dim=1))

        model = FaceClassifier()
    else:
        raise ValueError('Unsupported model architecture')
    model.pretrained_provenance = provenance if pretrained else {'source': 'RANDOM_WEIGHTS_TECHNICAL_TEST_ONLY'}
    return model


def set_training_stage(model, architecture, finetune):
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    head = model.fc if architecture == 'resnet18' else model.head
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    if finetune:
        encoder = model.layer4 if architecture == 'resnet18' else model.backbone
        for parameter in encoder.parameters():
            parameter.requires_grad_(True)
    frozen_eval(model)


def frozen_eval(model):
    """Keep frozen subtrees (including their BatchNorm statistics) in eval mode."""
    model.train()
    for module in model.modules():
        parameters = list(module.parameters())
        if parameters and not any(p.requires_grad for p in parameters):
            module.eval()
