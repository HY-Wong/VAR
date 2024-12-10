import os
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from PIL import Image


def pt_loader(path):
    with open(path, 'rb') as f:
        data = torch.load(f, weights_only=True)
    l1_hs = data['l1_hs']
    l2_hs = data['l2_hs']
    ll = data['ll']
    return l1_hs, l2_hs, ll


class WaveletDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def train_dataloader(self):
        train_set = DatasetFolder(
            root=os.path.join(self.args.data_path, 'train'),
            loader=pt_loader,
            extensions=('.pt',)
        )
        return DataLoader(
            dataset=train_set,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        val_set = DatasetFolder(
            root=os.path.join(self.args.data_path, 'val'),
            loader=pt_loader,
            extensions=('.pt',)
        )
        return DataLoader(
            dataset=val_set,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=False, drop_last=False
        )
    
    def test_dataloader(self):
        test_set = DatasetFolder(
            root=os.path.join(self.args.data_path, 'val'),
            loader=pt_loader,
            extensions=('.pt',)
        )
        return DataLoader(
            dataset=test_set,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=False, drop_last=False
        )


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, args, image_path: str, train_aug=None, val_aug=None):
        super().__init__()
        self.args = args
        self.image_path = image_path
        self.train_aug = train_aug
        self.val_aug = val_aug

    def train_dataloader(self):
        train_set = DatasetFolder(
            root=os.path.join(self.image_path, 'train'),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=self.train_aug
        )
        return DataLoader(
            dataset=train_set,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=True, drop_last=True
        )

    def val_dataloader(self):
        val_set = DatasetFolder(
            root=os.path.join(self.image_path, 'val'),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=self.val_aug
        )
        return DataLoader(
            dataset=val_set,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=False, drop_last=False
        )
    
    def test_dataloader(self):
        test_set = DatasetFolder(
            root=os.path.join(self.image_path, 'val'),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=self.val_aug
        )
        return DataLoader(
            dataset=test_set,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=False, drop_last=False
        )