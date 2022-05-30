import logging
import os
from argparse import ArgumentParser

from model import MyAwesomeModel, Pre_process
from PIL import Image

log = logging.getLogger(__name__)


def predict(args):
    """
    funtion to perform inference on the dataset
    """

    model = MyAwesomeModel.load_from_checkpoint(args.model_path)

    model.eval()
    transform = Pre_process()
    animals = [
        "dog",
        "horse",
        "elephant",
        "butterfly",
        "chicken",
        "cat",
        "cow",
        "sheep",
        "squirrel",
        "spider",
    ]

    for img in os.listdir(args.data_path):
        image = Image.open(img).convert("RGB")
        image = transform(image)
        preds = model(image)
        preds = int(preds.argmax())
        print("prediction ", animals[preds])

    log.info("Finish!!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--data_path", default=None)
    args = parser.parse_args()
    predict(args)
