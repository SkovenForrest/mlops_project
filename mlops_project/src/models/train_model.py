
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
log = logging.getLogger(__name__)


class AnimalDataset(Dataset):
    def __init__(self, labels, images_names,dir):
        self.img_labels = labels
        self.image_names = images_names
        self.dir = dir

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dir,self.image_names[idx])).convert('RGB')
        image = transform_images(image)
        label = self.img_labels[idx]   
        return image, label

def transform_images(data):
    transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5320, 0.5095, 0.4346), (0.2765, 0.2734, 0.2861))])
    return transform(data)

@hydra.main(config_path="config", config_name='default_config.yaml')
def train(config):

    log.info("Training day and night")
    hparams = config.experiment
    model_hparams = config.model
    torch.manual_seed(hparams["seed"])
    
    #parser = argparse.ArgumentParser(description='Training arguments')
    #parser.add_argument('--lr', default=0.1)
    # add any additional argument that you want
    #args = parser.parse_args(sys.argv[2:])
    #print(args)
    dir = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project\\data\\processed\\"
    dir_dataloader = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project"

    with open(dir+"test_img_list", "rb") as fp:   # Unpickling
        test_img_list = pickle.load(fp)

    with open(dir+"train_img_list", "rb") as fp:   # Unpickling
        train_img_list = pickle.load(fp)

    with open(dir+"test_targets", "rb") as fp:   # Unpickling
        test_targets = pickle.load(fp)

    with open(dir+"train_targets", "rb") as fp:   # Unpickling
        train_targets = pickle.load(fp)

    #train_images = torch.load(dir + "train_images.pt")
    #train_labels = torch.load(dir + "train_labels.pt")
    #test_images = torch.load(dir + "test_images.pt")
    #test_labels = torch.load(dir + "test_labels.pt")


    train_dataset = AnimalDataset(train_targets,train_img_list,dir_dataloader)
    test_dataset  = AnimalDataset(test_targets,test_img_list,dir_dataloader)

    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle= False, num_workers=6)
    
    # TODO: Implement training loop here
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

    trainer = Trainer(max_epochs=10, limit_train_batches=0.20, callbacks=[checkpoint_callback, early_stopping_callback],
            logger=pl.loggers.WandbLogger(project="mlops-final-project"),log_every_n_steps=50)

    trainer.fit(model,train_dataloader,test_dataloader)
    
    torch.save(model.state_dict(), "models/baseline.pth")

    log.info("Finish!!")

if __name__ == '__main__':
    train()
    