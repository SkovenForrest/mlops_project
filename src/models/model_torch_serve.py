import logging
import os
import pickle

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


log = logging.getLogger(__name__)


def transform_img(image):
    transforms_img = torch.nn.Sequential(transforms.Resize(128, 128), transforms.ToTensor())
    return transforms_img(image) / 255.0


class MyAwesomeModel(LightningModule):
    def __init__(self, configurations=None):
        super().__init__()

        self.model_resnet = models.resnet18(pretrained=True)
        num_ftrs = self.model_resnet.fc.in_features

        # Add fully connected layer for classification
        self.model_resnet.fc = nn.Linear(num_ftrs, 64)
        self.fc_out = nn.Linear(64, 10)

        self.criterium = nn.CrossEntropyLoss()
        if configurations is not None:
            self.random_crop = configurations["random_crop"]
            torch.manual_seed(self.configurations["seed"])
        else:
            self.random_crop = False

        self.configurations = configurations

        self.preprocess = transform_img

    def forward(self, x: Tensor):
        """ Forward pass through the network,
            the function returns the output logits
        """

        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 3 or x.shape[2] != 128 or x.shape[3] != 128:
            raise ValueError("Expected each sample to have shape [3, 128, 128]")

        x = self.model_resnet(x)
        x = F.relu(x)
        x = self.fc_out(x)
        return x

    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.configurations["lr"])

        return optimizer

    def train_dataloader(self):

        file_dir = os.path.dirname(os.path.abspath(__file__))
        head_tail = os.path.split(file_dir)
        head_tail2 = os.path.split(head_tail[0])
        base_path = head_tail2[0]
        dir_dataloader = base_path
        data_folder = os.path.join(dir_dataloader, "data")
        processed_folder = os.path.join(data_folder, "processed")
        dir = processed_folder

        with open(os.path.join(dir, "train_img_list"), "rb") as fp:  # Unpickling
            train_img_list = pickle.load(fp)

        with open(os.path.join(dir, "train_targets"), "rb") as fp:  # Unpickling
            train_targets = pickle.load(fp)

        train_dataset = AnimalDataset(
            train_targets, train_img_list, dir_dataloader, transform=self.preprocess
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=self.configurations["num_workers"],
        )

        return train_dataloader

    def val_dataloader(self):

        file_dir = os.path.dirname(os.path.abspath(__file__))
        head_tail = os.path.split(file_dir)
        head_tail2 = os.path.split(head_tail[0])
        base_path = head_tail2[0]
        dir_dataloader = base_path
        data_folder = os.path.join(dir_dataloader, "data")
        processed_folder = os.path.join(data_folder, "processed")
        dir = processed_folder

        with open(os.path.join(dir, "test_img_list"), "rb") as fp:  # Unpickling
            test_img_list = pickle.load(fp)

        with open(os.path.join(dir, "test_targets"), "rb") as fp:  # Unpickling
            test_targets = pickle.load(fp)

        test_dataset = AnimalDataset(
            test_targets, test_img_list, dir_dataloader, transform=self.preprocess
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=self.configurations["num_workers"],
        )

        return test_dataloader


class AnimalDataset(Dataset):
    """
    Dataset class for loading the animals dataset in pytorch format
    """

    def __init__(self, labels: list, images_names: list, dir: str, transform=None):
        self.img_labels = labels
        self.image_names = images_names
        self.dir = dir
        self.transforms = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        image = Image.open(os.path.join(self.dir, self.image_names[idx])).convert("RGB")
        if self.transforms is not None:
            image = self.transforms(image)
        # image = transform_images(image)
        label = self.img_labels[idx]
        return image, label
