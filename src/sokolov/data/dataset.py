"""Shared face crops; only model-specific size/normalization differs."""
from PIL import Image
from ..io import read_csv


def transforms(config, train=False):
    from torchvision import transforms as T
    steps = [T.Resize((config.model.image_size, config.model.image_size), antialias=True)]
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
