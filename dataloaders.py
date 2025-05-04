import os
import sys

import datasets
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor


def get_dataloader(args):
    if args.dataset == "set14":
        return Set14DataModule()
    elif args.dataset == "bsd100":
        return BSD100DataModule()
    elif args.dataset == "flickr8k":
        return Flickr8kDataModule()
    elif args.dataset == "div2k":
        return Div2KDataModule()
    if args.dataset == "CLIC2020":
        return CLIC(root=f"{args.data_root}/CLIC/2020")
    elif args.dataset == "CLIC2021":
        return CLIC(root=f"{args.data_root}/CLIC/2021")
    elif args.dataset == "Kodak":
        return Kodak(root=f"{args.data_root}/Kodak")
    elif args.dataset == "DIV2K":
        return DIV2K(root=f"{args.data_root}/DIV2K")
    else:
        print("Invalid dataset")
        sys.exit(0)


class Set14DataModule(LightningDataModule):
    def __init__(self, root=None, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        ds = datasets.load_dataset("eugenesiow/Set14", "bicubic_x2")
        self.val_dset = ds["validation"]
        self.test_dset = ds["validation"]

    def test_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )


class BSD100DataModule(LightningDataModule):
    def __init__(self, root=None, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        ds = datasets.load_dataset("eugenesiow/BSD100", "bicubic_x2")
        self.val_dset = ds["validation"]
        self.test_dset = ds["validation"]

    def test_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )


class Div2KDataModule(LightningDataModule):
    def __init__(self, root=None, batch_size=1):
        super().__init__()
        self.batch_size = batch_size

    def setup(self, stage=None):
        ds = datasets.load_dataset("eugenesiow/Div2k", "bicubic_x2")
        self.train_dset = ds["train"]
        self.val_dset = ds["validation"]
        self.test_dset = ds["validation"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )


class Flickr8kDataModule(LightningDataModule):
    def __init__(self, root="~/datasets/flickr8k", batch_size=1):
        super().__init__()
        self.root = os.path.expanduser(root)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.ds = datasets.load_dataset("atasoglu/flickr8k-dataset", data_dir=self.root)
        self.train_dset = self.ds["train"]
        self.val_dset = self.ds["validation"]
        self.test_dset = self.ds["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self):
        """实验需求，强制换成test set。"""
        return DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=False,
        )


class CLIC(LightningDataModule):
    def __init__(self, root, batch_size=1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        # self.train=train
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dset = ImageFolder(root=self.root + "/train", transform=transform)
        self.val_dset = ImageFolder(root=self.root + "/valid", transform=transform)
        self.test_dset = ImageFolder(root=self.root + "/test", transform=transform)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return loader


class DIV2K(LightningDataModule):
    def __init__(self, root, batch_size=1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        # self.train=train
        transform = transforms.Compose([transforms.ToTensor()])
        self.train_dset = ImageFolder(root=self.root + "/train", transform=transform)
        self.test_dset = ImageFolder(root=self.root + "/val", transform=transform)

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return loader


class Kodak(LightningDataModule):
    def __init__(self, root, batch_size=1):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        # self.train=train
        transform = transforms.Compose([transforms.ToTensor()])
        self.test_dset = ImageFolder(root=self.root, transform=transform)

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dset,
            batch_size=self.batch_size,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )
        return loader


# class ImageDataset(Dataset):
#     def __init__(self, image_paths, transform=None):
#         self.image_paths = image_paths
#         self.transform = transform

#     def get_class_label(self, image_name):
#         # your method here
#         y = ...
#         return y

#     def __getitem__(self, index):
#         image_path = self.image_paths[index]
#         x = Image.open(image_path)
#         y = self.get_class_label(image_path.split('/')[-1])
#         if self.transform is not None:
#             x = self.transform(x)
#         return x, y

#     def __len__(self):
#         return len(self.image_paths)
