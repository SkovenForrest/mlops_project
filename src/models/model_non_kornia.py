import logging
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor, nn

log = logging.getLogger(__name__)


class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=10, kernel_size=5, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=10, out_channels=10, kernel_size=5, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=10, out_channels=20, kernel_size=5, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=20, out_channels=30, kernel_size=5, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=30, out_channels=30, kernel_size=5, stride=1, padding=1
        )

        self.bn1 = nn.BatchNorm2d(10)
        self.bn2 = nn.BatchNorm2d(20)
        self.bn3 = nn.BatchNorm2d(30)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2430, 10)

        self.criterium = nn.CrossEntropyLoss()

    def forward(self, x: Tensor):
        """ Forward pass through the network, 
            the function returns the output logits
        """
        if x.ndim != 4:
            raise ValueError("Expected input to a 4D tensor")
        if x.shape[1] != 3 or x.shape[2] != 64 or x.shape[3] != 64:
            raise ValueError("Expected each sample to have shape [3, 64, 64]")

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv3(x)))
        x = F.relu(self.bn2(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv5(x)))
        x = F.relu(self.bn3(self.conv6(x)))
        x = torch.flatten(x, 1)

        x = self.fc1(x)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer
