import logging
from model import MyAwesomeModel, Pre_process
from argparse import ArgumentParser
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

    image = Image.open(args.image_path).convert("RGB")
    image = transform(image)
    image = image[None, :]
    preds = model(image)
    preds = int(preds.argmax())
    print("prediction: ", animals[preds])

    log.info("Finish!!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_path", default=None, type=str, help="the path to the model"
    )
    parser.add_argument(
        "--image_path",
        default=None,
        type=str,
        help="path to the image to preform predictions on ",
    )
    args = parser.parse_args()
    predict(args)
