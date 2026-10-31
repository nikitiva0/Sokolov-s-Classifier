"""Shared face crops; only model-specific size/normalization differs.

Train-only augmentation draws every random number from torch's RNG, so the
per-epoch seed (main process or DataLoader worker seed) fixes it exactly.
"""
from PIL import Image
from ..io import read_csv


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


def train_augmentation(config):
    """Geometric, resolution/compression and colour perturbations for both classes."""
    from torchvision import transforms as T
    t, size = config.training, config.model.image_size
    jitter = t.aug_color_jitter
    return [T.RandomResizedCrop(size, scale=(t.aug_crop_scale_min, 1.), ratio=(.9, 1/.9), antialias=True),
            T.RandomRotation(t.aug_rotation_deg),
            T.RandomHorizontalFlip(),
            RandomDownscale(t.aug_downscale_min, t.aug_downscale_max, t.aug_downscale_p),
            RandomJPEG(t.aug_jpeg_min, t.aug_jpeg_max, t.aug_jpeg_p),
            T.RandomApply([T.GaussianBlur(5, sigma=(.1, 2.))], p=t.aug_blur_p),
            T.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter),
            T.RandomGrayscale(t.aug_grayscale_p)]


def transforms(config, train=False):
    from torchvision import transforms as T
    size = config.model.image_size
    if train and config.training.augment:
        steps = train_augmentation(config)
    else:
        steps = [T.Resize((size, size), antialias=True)]
        if train:
            steps += [T.RandomHorizontalFlip(), T.ColorJitter(brightness=.1, contrast=.1, saturation=.1)]
    if config.model.architecture == 'resnet18':
        mean, std = (.485, .456, .406), (.229, .224, .225)
    else:
        mean = std = (.5, .5, .5)
    return T.Compose(steps + [T.ToTensor(), T.Normalize(mean, std)])


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
