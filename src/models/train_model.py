
from model import  MyAwesomeModel

import torch
import hydra
from omegaconf import OmegaConf
import logging
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
from PIL import Image
import pickle
import os
import kornia 
from torch import nn
from torchvision.transforms import Resize
import numpy as np
log = logging.getLogger(__name__)


"""
def transform_train(data):
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5320, 0.5095, 0.4346), (0.2765, 0.2734, 0.2861)),transforms.RandomHorizontalFlip(p=0.5),transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)])
    return transform(data)

def transform_test(data):
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5320, 0.5095, 0.4346), (0.2765, 0.2734, 0.2861))])
    return transform(data)
"""


@hydra.main(config_path="config", config_name='default_config.yaml')
def train(config):

    log.info("Training day and night")
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])

    model = MyAwesomeModel()
    print(model)

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor="val_loss", mode="min"
        )

    early_stopping_callback = EarlyStopping(
        monitor="val_loss", patience=5, verbose=True, mode="min"
        )

   # trainer = Trainer(max_epochs=10, limit_train_batches=0.20, callbacks=[checkpoint_callback, early_stopping_callback],
   #         logger=pl.loggers.WandbLogger(project="mlops-final-project"), profiler="simple",log_every_n_steps=10)

    #trainer = Trainer(max_epochs=10, limit_train_batches=0.20, callbacks=[checkpoint_callback, early_stopping_callback],
    #        logger=pl.loggers.WandbLogger(project="mlops-final-project"),log_every_n_steps=50)

    trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback, early_stopping_callback],
            logger=pl.loggers.WandbLogger(project="mlops-final-project"),log_every_n_steps=50)
    trainer.fit(model)
    
    torch.save(model.state_dict(), "models/baseline.pth")

    log.info("Finish!!")

if __name__ == '__main__':
    train()
    