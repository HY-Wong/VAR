import os
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
from PIL import Image


def normalize_01_into_pn1(x):
    """
    Normalize x from [0, 1] to [-1, 1]
    """
    return x.add(x).add_(-1)


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
    return img


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        final_reso = 256
        mid_reso = round(1.125 * final_reso)
        train_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.RandomCrop((final_reso, final_reso)),
            transforms.ToTensor(), 
            normalize_01_into_pn1
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        val_aug = [
            transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS),
            transforms.CenterCrop((final_reso, final_reso)),
            transforms.ToTensor(), 
            normalize_01_into_pn1
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        self.train_aug = transforms.Compose(train_aug)
        self.val_aug = transforms.Compose(val_aug)
        
    def train_dataloader(self):
        train_set = DatasetFolder(
            root=os.path.join(self.args.data_path, 'train'),
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
            root=os.path.join(self.args.data_path, 'val'),
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
            root=os.path.join(self.args.data_path, 'val'),
            loader=pil_loader,
            extensions=IMG_EXTENSIONS,
            transform=self.val_aug
        )
        return DataLoader(
            dataset=test_set,
            batch_size=self.args.batch_size, num_workers=self.args.workers,
            shuffle=False, drop_last=False
        )