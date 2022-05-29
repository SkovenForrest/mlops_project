import logging
import os
import pickle
import hydra
import pytorch_lightning as pl
import torch
from model import MyAwesomeModel
from PIL import Image
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

log = logging.getLogger(__name__)


class AnimalDataset(Dataset):
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


def transform_train(data):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5320, 0.5095, 0.4346), (0.2765, 0.2734, 0.2861)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
            ),
        ]
    )
    return transform(data)


def transform_test(data):
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5320, 0.5095, 0.4346), (0.2765, 0.2734, 0.2861)),
        ]
    )
    return transform(data)


@hydra.main(config_path="config", config_name="default_config.yaml")
def train(config):

    log.info("Training day and night")
    hparams = config.experiment
    torch.manual_seed(hparams["seed"])

    dir = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project\\data\\processed\\"
    dir_dataloader = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project"

    with open(dir + "test_img_list", "rb") as fp:  # Unpickling
        test_img_list = pickle.load(fp)

    with open(dir + "train_img_list", "rb") as fp:  # Unpickling
        train_img_list = pickle.load(fp)

    with open(dir + "test_targets", "rb") as fp:  # Unpickling
        test_targets = pickle.load(fp)

    with open(dir + "train_targets", "rb") as fp:  # Unpickling
        train_targets = pickle.load(fp)

    train_dataset = AnimalDataset(
        train_targets, train_img_list, dir_dataloader, transform=transform_train
    )
    test_dataset = AnimalDataset(
        test_targets, test_img_list, dir_dataloader, transform=transform_test
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=6
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=6
    )

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

    trainer = Trainer(
        max_epochs=10,
        limit_train_batches=0.20,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=pl.loggers.WandbLogger(project="mlops-final-project"),
        log_every_n_steps=50,
    )

    trainer.fit(model, train_dataloader, test_dataloader)

    torch.save(model.state_dict(), "models/baseline.pth")

    log.info("Finish!!")


if __name__ == "__main__":
    train()
