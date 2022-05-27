import torch
from torch import Tensor, nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
import logging
import kornia as K
import numpy as np
from PIL import Image
import torchmetrics
import pickle
import os
from torch.utils.data import DataLoader, Dataset

log = logging.getLogger(__name__)
class MyAwesomeModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=1)


        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(96)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(2048, 40)
        self.fc2 = nn.Linear(40,10)

        self.criterium = nn.CrossEntropyLoss()

        self.preprocess = Pre_process()

        self.transform = Data_augmentation(apply_color_jitter=True)



    def forward(self, x: Tensor):
        ''' Forward pass through the network, 
            the function returns the output logits
        '''

        if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
        if x.shape[1] != 3 or x.shape[2] != 64 or x.shape[3] != 64:
            raise ValueError('Expected each sample to have shape [3, 64, 64]')

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)       
        x = F.relu(self.bn2(self.conv2(x))) 
        x = self.pool(x)                       
        x = F.relu(self.bn3(self.conv3(x)))  
        x = self.pool(x)   
        x = F.relu(self.bn4(self.conv4(x)))   
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.fc2(x)
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
        self.log("train_loss",loss)
        self.log("train_acc",acc)
        return loss
    
    def validation_step(self, batch, batch_idx):
        data, target = batch
        preds = self(data)
        loss = self.criterium(preds, target)
        acc = (target == preds.argmax(dim=-1)).float().mean()
        self.log("val_loss",loss)
        self.log("val_acc", acc)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        return optimizer

    def train_dataloader(self):
        dir = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project\\data\\processed\\"
        dir_dataloader = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project"

        with open(dir+"train_img_list", "rb") as fp:   # Unpickling
            train_img_list = pickle.load(fp)

        with open(dir+"train_targets", "rb") as fp:   # Unpickling
            train_targets = pickle.load(fp)

        train_dataset = AnimalDataset(train_targets,train_img_list,dir_dataloader,transform=self.preprocess)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)

        return train_dataloader

    def val_dataloader(self):
        dir = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project\\data\\processed\\"
        dir_dataloader = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project"

        with open(dir+"test_img_list", "rb") as fp:   # Unpickling
            test_img_list = pickle.load(fp)

        with open(dir+"test_targets", "rb") as fp:   # Unpickling
            test_targets = pickle.load(fp)

        test_dataset  = AnimalDataset(test_targets,test_img_list,dir_dataloader, transform= self.preprocess)
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle= False, num_workers=6)

        return test_dataloader
       


class Data_augmentation(nn.Module):
    """ 
    perform data augmentation on torch tensors using Kornia.
    """
    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transform = nn.Sequential(
            K.augmentation.RandomHorizontalFlip(p=0.5)
        )

        self.jitter = K.augmentation.ColorJitter(0.1, 0.1, 0.1, 0.1)


    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_out = self.transform(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out


class Pre_process(nn.Module):
    """Module to perform pre-process using Kornia on torch tensors."""
    def __init__(self) -> None:
        super().__init__()
 
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Image) -> torch.Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_tensor: torch.Tensor = K.image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        x_resize: torch.Tensor = K.augmentation.Resize((64, 64))(x_tensor.float())
        x_norm: torch.Tensor = K.augmentation.Normalize(torch.Tensor([0.5320, 0.5095, 0.4346]), torch.Tensor([0.2765, 0.2734, 0.2861]))(x_resize.float())
        x_out: torch.Tensor = torch.squeeze(x_norm)
        return x_out.float()



class AnimalDataset(Dataset):
    def __init__(self, labels: list, images_names: list, dir: str, transform=None):
        self.img_labels = labels
        self.image_names = images_names
        self.dir = dir
        self.transforms = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx: int):
        image = Image.open(os.path.join(self.dir,self.image_names[idx])).convert('RGB')
        if self.transforms is not None:
            image = self.transforms(image)
        #image = transform_images(image)
        label = self.img_labels[idx]   
        return image, label


