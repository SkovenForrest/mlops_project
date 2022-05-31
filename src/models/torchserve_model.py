import logging
from model_torch_serve import MyAwesomeModel
from argparse import ArgumentParser
import torch

log = logging.getLogger(__name__)

def serve_model(args):
    """
    funtion to perform inference on the dataset
    """

    model = MyAwesomeModel.load_from_checkpoint(args.model_path)

    scripted_model = model.to_torchscript()
    torch.jit.save(scripted_model,'deployable_model.pt')    
    log.info("Finish!!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default=None, type=str, help='the path to the model')
    args = parser.parse_args()
    serve_model(args)
