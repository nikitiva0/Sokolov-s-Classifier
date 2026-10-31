"""Differentiable face-only normalization shared by training, UI and synthesis."""
def face_mask(pixels):
    import torch
    height, width = pixels.shape[-2:]
    y = torch.linspace(-1, 1, height, device=pixels.device, dtype=pixels.dtype)[:, None]
    x = torch.linspace(-1, 1, width, device=pixels.device, dtype=pixels.dtype)[None, :]
    # Canonical five-point alignment places eyes near y=.46, mouth near y=.82.
    # Keep the inner face and beard, suppress the scene in upper/lateral corners.
    return (((x/.86)**2 + ((y-.18)/1.02)**2) <= 1).to(pixels.dtype)


def preprocess_face(pixels, policy='rgb'):
    if policy == 'rgb':
        return pixels
    if policy != 'identity_v1':
        raise ValueError('Unknown face preprocessing policy')
    if pixels.ndim not in (3, 4) or pixels.shape[-3] != 3:
        raise ValueError('Expected RGB CHW or NCHW tensor')
    gray = (pixels * pixels.new_tensor([.299, .587, .114]).reshape(
        *((1,) if pixels.ndim == 4 else ()), 3, 1, 1)).sum(-3, keepdim=True)
    mask = face_mask(pixels)
    count = mask.sum()
    mean = (gray*mask).sum((-2, -1), keepdim=True)/count
    variance = ((gray-mean).square()*mask).sum((-2, -1), keepdim=True)/count
    normalized = ((gray-mean)*(.18/variance.clamp_min(.05**2).sqrt())+.5).clamp(0, 1)
    output = normalized*mask + .5*(1-mask)
    # Put both classes on the same effective pixel grid. Otherwise a low-res
    # webcam face vs a sharp FairFace portrait can identify the source alone.
    import torch.nn.functional as F
    original_size = output.shape[-2:]
    batched = output if output.ndim == 4 else output.unsqueeze(0)
    if original_size[0] == original_size[1] and original_size[0] in (112, 224):
        step = original_size[0]//56
        small = F.avg_pool2d(batched, step, step)
    else:
        small = F.interpolate(batched, (56, 56), mode='bilinear', align_corners=False)
    output = F.interpolate(small, original_size, mode='bilinear', align_corners=False)
    if pixels.ndim == 3:
        output = output[0]
    output = output*mask + .5*(1-mask)
    return output.expand_as(pixels)


def normalized_face(pixels, architecture, policy='rgb'):
    pixels = preprocess_face(pixels, policy)
    if architecture == 'resnet18':
        mean, std = (.485, .456, .406), (.229, .224, .225)
    elif architecture == 'arcface':
        mean = std = (.5, .5, .5)
    else:
        raise ValueError('Unknown architecture')
    shape = (1, 3, 1, 1) if pixels.ndim == 4 else (3, 1, 1)
    return (pixels-pixels.new_tensor(mean).view(shape))/pixels.new_tensor(std).view(shape)
