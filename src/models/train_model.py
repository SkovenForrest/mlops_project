import logging
import hydra
import pytorch_lightning as pl
import torch
from model import MyAwesomeModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="default_config.yaml")
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

    # trainer = Trainer(max_epochs=10, limit_train_batches=0.20, callbacks=[checkpoint_callback, early_stopping_callback],
    #        logger=pl.loggers.WandbLogger(project="mlops-final-project"),log_every_n_steps=50)

    trainer = Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback, early_stopping_callback],
        logger=pl.loggers.WandbLogger(project="mlops-final-project"),
        log_every_n_steps=50,
    )
    trainer.fit(model)

    torch.save(model.state_dict(), "models/baseline.pth")

    log.info("Finish!!")


if __name__ == "__main__":
    train()
