import pytest
import torch
from src.models.model import MyAwesomeModel


class Test_model:
    @pytest.mark.parametrize(
        "input,expected_output",
        [
            (torch.rand((64, 3, 128, 128)), torch.Size([64, 10])),
            (torch.rand((32, 3, 128, 128)), torch.Size([32, 10])),
        ],
    )
    def test_output_shape(self, input, expected_output):
        model = MyAwesomeModel()
        output = model(input)
        assert output.shape == expected_output
