"""Shared face crops; only model-specific size/normalization differs.

Train-only augmentation draws every random number from torch's RNG, so the
per-epoch seed (main process or DataLoader worker seed) fixes it exactly.
Validation, test, UI and generation use one deterministic resize + face
preprocessing (model.face_preprocessing) + normalization.
"""
import math

from PIL import Image
from ..io import read_csv
from .families import source_family


def _uniform_int(low, high):
    import torch
    return int(torch.randint(low, high + 1, ()).item())


def _chance(p):
    import torch
    return p > 0 and torch.rand(()).item() < p


class RandomDownscale:
    """Resize the face to a random side U[low, high] px and back (small-face/webcam simulation)."""

    def __init__(self, low, high, p):
        self.low, self.high, self.p = low, high, p

    def __call__(self, image):
        if not _chance(self.p):
            return image
        side = _uniform_int(self.low, self.high)
        if side >= min(image.size):
            return image
        return image.resize((side, side), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)

    def __repr__(self):
        return f'RandomDownscale(side={self.low}..{self.high}px, p={self.p})'


class RandomJPEG:
    """Re-encode with a random JPEG quality in [low, high]."""

    def __init__(self, low, high, p):
        self.low, self.high, self.p = low, high, p

    def __call__(self, image):
        from io import BytesIO
        if not _chance(self.p):
            return image
        buffer = BytesIO()
        image.save(buffer, 'JPEG', quality=_uniform_int(self.low, self.high))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert('RGB')

    def __repr__(self):
        return f'RandomJPEG(quality={self.low}..{self.high}, p={self.p})'


class NormalizeFace:
    """Deterministic face preprocessing (preprocessing.py) + model normalization on a CHW tensor."""

    def __init__(self, architecture, policy):
        self.architecture, self.policy = architecture, policy

    def __call__(self, pixels):
        from ..preprocessing import normalized_face
        return normalized_face(pixels, self.architecture, self.policy)

    def __repr__(self):
        return f'NormalizeFace(architecture={self.architecture}, policy={self.policy})'


def robust_augmentation(config):
    """training.augmentation="robust": one recipe for both classes, parameters training.aug_*.

    Order: geometry (crop/flip/affine) -> colour -> resolution (downscale, blur) -> JPEG.
    Steps with zero strength/probability are omitted, so they draw no random numbers.
    """
    from torchvision import transforms as T
    t, size = config.training, config.model.image_size
    steps = ([T.RandomResizedCrop(size, scale=(t.aug_crop_scale_min, 1.), ratio=(.9, 1/.9), antialias=True)]
             if t.aug_crop_scale_min < 1 else [T.Resize((size, size), antialias=True)])
    steps.append(T.RandomHorizontalFlip())
    scale = (t.aug_scale_min, t.aug_scale_max) if (t.aug_scale_min, t.aug_scale_max) != (1., 1.) else None
    translate = (t.aug_translate, t.aug_translate) if t.aug_translate else None
    if t.aug_rotation_deg or translate or scale:
        steps.append(T.RandomAffine(degrees=t.aug_rotation_deg, translate=translate, scale=scale, fill=128))
    if t.aug_color_jitter or t.aug_saturation or t.aug_hue:
        steps.append(T.ColorJitter(brightness=t.aug_color_jitter, contrast=t.aug_color_jitter,
                                   saturation=t.aug_saturation, hue=t.aug_hue))
    if t.aug_grayscale_p:
        steps.append(T.RandomGrayscale(t.aug_grayscale_p))
    if t.aug_downscale_p:
        steps.append(RandomDownscale(t.aug_downscale_min, t.aug_downscale_max, t.aug_downscale_p))
    if t.aug_blur_p:
        kernel = 2*math.ceil(3*t.aug_blur_sigma_max)+1
        steps.append(T.RandomApply([T.GaussianBlur(kernel, sigma=(t.aug_blur_sigma_min, t.aug_blur_sigma_max))],
                                   p=t.aug_blur_p))
    if t.aug_jpeg_p:
        steps.append(RandomJPEG(t.aug_jpeg_min, t.aug_jpeg_max, t.aug_jpeg_p))
    return steps


def transforms(config, train=False):
    from torchvision import transforms as T
    size = config.model.image_size
    if train and config.training.augmentation == 'robust':
        steps = robust_augmentation(config)
    else:
        steps = [T.Resize((size, size), antialias=True)]
        if train:
            steps += [T.RandomHorizontalFlip(), T.ColorJitter(brightness=.1, contrast=.1, saturation=.1)]
    return T.Compose(steps + [T.ToTensor(), NormalizeFace(
        config.model.architecture, config.model.face_preprocessing)])


def source_sampling_weights(rows):
    """Per-row sampling weights: equal mass per acquisition family, then per source, within a class.

    Families come from data.families (lecture / web / fairface). O(N).
    """
    from collections import Counter
    counts = Counter((int(r['label']), source_family(r), r['source_id']) for r in rows)
    classes = Counter(int(r['label']) for r in rows)
    source_counts = Counter((label, family) for label, family, _ in counts)
    families = Counter(label for label, _ in source_counts)
    # Balance acquisition families, then sources within a family. Total class mass
    # still equals its train count, so train-only BCE pos_weight stays appropriate.
    return [classes[int(r['label'])]/families[int(r['label'])]/
            source_counts[(int(r['label']), source_family(r))]/
            counts[(int(r['label']), source_family(r), r['source_id'])] for r in rows]


class FaceDataset:
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import torch
        row = self.rows[index]
        with Image.open(row['crop_path']) as im:
            image = self.transform(im.convert('RGB'))
        return image, torch.tensor(float(row['label'])), row['image_id']


def build_dataset(config, split):
    rows = [r for r in read_csv(config.data.prepared_manifest) if r['split'] == split and r['status'] == 'ok']
    if not rows:
        raise ValueError(f'No prepared faces in {split}')
    return FaceDataset(rows, transforms(config, train=split == 'train'))
