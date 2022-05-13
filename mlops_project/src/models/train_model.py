
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


log = logging.getLogger(__name__)




class AnimalDataset(Dataset):
    def __init__(self, labels, images):
        self.img_labels = labels
        self.img = images

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image =self.img[idx]
        image = transform_images(image)
        label = self.img_labels[idx]   
        return image, label


def transform_images(data):
    transform = transforms.Compose([transforms.Resize((128,128))])
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
    train_images = torch.load(dir + "train_images.pt")
    train_labels = torch.load(dir + "train_labels.pt")
    test_images = torch.load(dir + "test_images.pt")
    test_labels = torch.load(dir + "test_labels.pt")


    train_dataset = AnimalDataset(train_labels,train_images)
    test_dataset  = AnimalDataset(test_labels,test_images)

    train_dataloader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=6)
    test_dataloader = DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle= False, num_workers=6)
    
    # TODO: Implement training loop here
    model = MyAwesomeModel(model_hparams)
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

def test_score(test_set,model,criterion):
    
    running_loss = 0.0
    accuracy = 0.0
    total = 0
    correct = 0
    with torch.no_grad():

        for i, data in enumerate(test_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            classes = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (classes == labels).sum().item()
            running_loss += loss.item()

        accuracy = 100.*correct/total
        test_loss = running_loss/len(test_set)
    return accuracy, test_loss


if __name__ == '__main__':
    train()
    