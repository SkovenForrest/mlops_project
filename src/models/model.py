import logging
import os
import pickle

import kornia as K
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

log = logging.getLogger(__name__)


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

        self.preprocess = Pre_process(random_crop=self.random_crop)

        self.transform = Data_augmentation(configurations=self.configurations)

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

    def on_after_batch_transfer(self, batch, dataloader_idx):
        x, y = batch
        if self.trainer.training:
            x = self.transform(x)  # => we perform GPU/Batched data augmentation
        return x, y

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


class Data_augmentation(nn.Module):
    """
    perform data augmentation on torch tensors using Kornia.
    the augmentations can be controlled from the configuration files
    """

    def __init__(self, configurations: dict) -> None:
        super().__init__()
        if configurations is not None:
            self.configurations = configurations
            brightness = self.configurations["brightness"]
            contrast = self.configurations["contrast"]
            saturation = self.configurations["saturation"]
            hue = self.configurations["saturation"]
            mean = self.configurations["noise_mean"]
            std = self.configurations["noise_std"]

            self.horizontal_flip = K.augmentation.RandomHorizontalFlip(
                p=self.configurations["p_horizontal_flip"]
            )
            self.jitter = K.augmentation.ColorJitter(
                brightness, contrast, saturation, hue, p=self.configurations["p_jitter"]
            )
            self.rotate = K.augmentation.RandomRotation(
                self.configurations["degrees"], p=self.configurations["p_rotate"]
            )
            self.gaussian_noise = K.augmentation.RandomGaussianNoise(
                mean=mean, std=std, p=self.configurations["p_noise"]
            )

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.configurations is not None:
            if self.configurations["apply_horizontal_flip"]:
                x = self.horizontal_flip(x)

            if self.configurations["apply_color_jitter"]:
                x = self.jitter(x)

            if self.configurations["apply_rotate"]:
                x = self.rotate(x)

            if self.configurations["apply_gaussian_noise"]:
                x = self.gaussian_noise(x)

        return x


class Pre_process(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""

    def __init__(self, random_crop: bool = False) -> None:
        super().__init__()
        self.random_crop = random_crop

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_tensor: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        if self.random_crop:
            x_resize: torch.Tensor = K.augmentation.Resize((128, 128))(x_tensor.float())
        else:
            x_resize: torch.Tensor = K.augmentation.RandomCrop(
                (128, 128), pad_if_needed=True
            )(x_tensor.float())
        x_out: torch.Tensor = torch.squeeze(x_resize)
        return x_out.float() / 255.0


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
