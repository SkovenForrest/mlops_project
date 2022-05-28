import logging

import hydra
import torch
import torch.nn.functional as F
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from torch import nn

log = logging.getLogger(__name__)
class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        """
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64,  3)
        self.conv3 = nn.Conv2d(64, 128,  3)
        self.conv4 = nn.Conv2d(128, 256,  3)
        self.fc1 = nn.Linear(1024,  40)
        self.fc2 = nn.Linear(40, 10)
        """
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=20, out_channels=30, kernel_size=5, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=30, out_channels=30, kernel_size=5, stride=1, padding=1)

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(30)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(2430, 10)

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x):
        """
        if x.ndim != 4:
          raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 3 or x.shape[2] != 64 or x.shape[3] != 64:
            raise ValueError('Expected each sample to have shape [3, 128, 128]')

        #log.info("x.shape input",x.shape)  # shape ( batch_size, 3, 128, 128)
        x = self.pool(F.relu(self.conv1(x)))  # shape = (batch_size, 4, 62, 62)
        #log.info("x.shape after conv1",x.shape)
        x = self.pool(F.relu(self.conv2(x))) # shape = (batch_size, 6, 29, 29)
        #log.info("x.shape after conv2",x.shape)
        x = self.pool(F.relu(self.conv3(x))) # shape = (batch_size, 8, 12, 12)
        #log.info("x.shape after conv3",x.shape)
        x = self.pool(F.relu(self.conv4(x))) # shape = (batch_size, 10, 4, 4)
        #log.info("x.shape after conv4",x.shape)
        x = torch.flatten(x, 1) # flatten all dimensions except batch   # shape = (batch_size, 240)
        #log.info("x.shape after flatten",x.shape)
        x = F.relu(self.fc1(x))
        #log.info("x.shape after fc1",x.shape)
        x = self.fc2(x)
        """
        return x
    
    def training_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("train_loss",loss)
        self.log("train_acc",acc)
        self.logger.experiment.log({'train_logits': wandb.Histogram(preds.detach().numpy())})
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        print("preds", type(preds))
        print("targets", type(target))
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss",loss)
        self.log("val_acc",acc)
        self.logger.experiment.log({'val_logits': wandb.Histogram(preds.detach().numpy())})
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer
       

