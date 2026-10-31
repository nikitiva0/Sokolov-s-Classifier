"""Shared face crops; only model-specific size/normalization differs."""
from PIL import Image
from ..io import read_csv


class ResolutionAugmentation:
    """The same blur/compression distribution for both classes, independent of label."""
    def __call__(self, image):
        import random
        from io import BytesIO
        from PIL import ImageFilter
        if random.random() < .75:
            side = random.choice([32, 40, 48, 64, 80, 112, 160, 224])
            image = image.resize((side, side), Image.Resampling.BILINEAR).resize(
                image.size, Image.Resampling.BILINEAR)
        if random.random() < .3:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(.2, 1.2)))
        if random.random() < .4:
            stream = BytesIO()
            image.save(stream, format='JPEG', quality=random.randint(35, 95))
            stream.seek(0)
            with Image.open(stream) as compressed:
                image = compressed.convert('RGB')
        return image


class NormalizeFace:
    def __init__(self, architecture, policy):
        self.architecture, self.policy = architecture, policy

    def __call__(self, pixels):
        from ..preprocessing import normalized_face
        return normalized_face(pixels, self.architecture, self.policy)


def transforms(config, train=False):
    from torchvision import transforms as T
    steps = [T.Resize((config.model.image_size, config.model.image_size), antialias=True)]
    if train:
        steps += [T.RandomHorizontalFlip()]
        if config.training.augmentation == 'robust':
            steps += [T.RandomAffine(degrees=6, translate=(.025, .025), scale=(.95, 1.05), fill=128),
                      T.ColorJitter(brightness=.35, contrast=.35, saturation=.3, hue=.04),
                      ResolutionAugmentation()]
        else:
            steps += [T.ColorJitter(brightness=.1, contrast=.1, saturation=.1)]
    return T.Compose(steps + [T.ToTensor(), NormalizeFace(
        config.model.architecture, config.model.face_preprocessing)])


def source_sampling_weights(rows):
    from collections import Counter
    def family(row):
        source = row['source_id']
        return 'web' if source.startswith('web:') else 'fairface' if source.startswith('fairface:') else 'lecture'
    counts = Counter((int(r['label']), family(r), r['source_id']) for r in rows)
    classes = Counter(int(r['label']) for r in rows)
    source_counts = Counter((label, domain) for label, domain, _ in counts)
    domains = Counter(label for label, _ in source_counts)
    # Balance acquisition domains, then sources within a domain. Total class mass
    # still equals its train count, so train-only BCE pos_weight stays appropriate.
    return [classes[int(r['label'])]/domains[int(r['label'])]/
            source_counts[(int(r['label']), family(r))]/
            counts[(int(r['label']), family(r), r['source_id'])] for r in rows]


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
