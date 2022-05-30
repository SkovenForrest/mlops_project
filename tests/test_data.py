import os.path

import numpy as np
import pytest

from src.models.model import MyAwesomeModel

model = MyAwesomeModel()
loader = model.train_dataloader
loader2 = model.train_dataloader()


train_loader = model.train_dataloader()
val_loader = model.val_dataloader()


class Test_data:
    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_train_images_length(self):
        assert len(train_loader.dataset.image_names) == 20943, "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_val_images_length(self):
        assert len(val_loader.dataset.image_names) == 5236, "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_train_labels_length(self):
        assert len(train_loader.dataset.img_labels) == 20943, "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_val_labels_length(self):
        assert len(val_loader.dataset.img_labels) == 5236, "not enough datasamples"

    @pytest.mark.skipif(not os.path.exists("processed"), reason="Data files not found")
    def test_labels_shape(self):
        assert all(
            np.unique(train_loader.dataset.img_labels) == np.arange(0, 10)
        ), "not all 10 classes present"
        assert all(
            np.unique(val_loader.dataset.img_labels) == np.arange(0, 10)
        ), "not all 10 classes present"
