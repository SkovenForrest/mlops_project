
import logging

import hydra
import kornia
import numpy as np
import pytorch_lightning as pl
import torch
from model import MyAwesomeModel
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.transforms import Resize


def transform_test(data):
    transform = nn.Sequential(
    kornia.geometry.Resize(64,64),
    kornia.image_to_tensor(data),
    kornia.enhance.normalize(data,mean=torch.tensor([0.5320, 0.5095, 0.4346]), std=torch.tensor([0.2765, 0.2734, 0.2861])),
    )
    return transform



image = Image.open("data\\processed\\test\\2.jpeg").convert('RGB')

image = np.array(image)
print("before",image.shape)
print("ten",kornia.image_to_tensor(image).shape)
image = transform_test(image)
print("after",image.shape)

