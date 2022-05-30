import os.path

import hydra
import numpy as np
import pytest
import torch

from src.models.model import MyAwesomeModel


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.1")
class Test_data:
    def __init__(self, config):
        super().__init__()

        self.hparams = config.experiment
        torch.manual_seed(self.hparams["seed"])

        model = MyAwesomeModel(configurations=self.hparams)

        self.train_loader = model.train_dataloader()
        self.val_loader = model.val_dataloader()

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_train_images_length(self):
        assert (
            len(self.train_loader.dataset.image_names) == 20943
        ), "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_val_images_length(self):
        assert (
            len(self.val_loader.dataset.image_names) == 5236
        ), "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_train_labels_length(self):
        assert (
            len(self.train_loader.dataset.img_labels) == 20943
        ), "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_val_labels_length(self):
        assert len(self.val_loader.dataset.img_labels) == 5236, "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_labels_shape(self):
        assert all(
            np.unique(self.train_loader.dataset.img_labels) == np.arange(0, 10)
        ), "not all 10 classes present"
        assert all(
            np.unique(self.val_loader.dataset.img_labels) == np.arange(0, 10)
        ), "not all 10 classes present"
