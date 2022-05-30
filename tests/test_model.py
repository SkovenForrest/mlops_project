import hydra
import pytest
import torch

from src.models.model import MyAwesomeModel


@hydra.main(config_path="config", config_name="default_config.yaml", version_base="1.1")
class Test_data:
    def __init__(self, config):
        super().__init__()

        self.hparams = config.experiment
        torch.manual_seed(self.hparams["seed"])

        self.model = MyAwesomeModel(configurations=self.hparams)

    @pytest.mark.parametrize(
        "input,expected_output",
        [
            (torch.rand((64, 3, 128, 128)), torch.Size([64, 10])),
            (torch.rand((32, 3, 128, 128)), torch.Size([32, 10])),
        ],
    )
    def test_output_shape(self, input, expected_output):
        output = self.model(input)
        assert output.shape == expected_output
