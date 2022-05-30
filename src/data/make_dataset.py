# -*- coding: utf-8 -*-
import logging
import os
import pickle
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv
from PIL import Image
from sklearn.model_selection import train_test_split


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath: str, output_filepath: str):
    """
    Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # dataset from https://www.kaggle.com/datasets/alessiocorrado99/animals10

    # terminal command python src/data/make_dataset.py data/raw/ data/processed/

    logger.info("loading the images")
    dataset = load_dataset(input_filepath)

    logger.info("making the train and test split")
    x_train, x_test, y_train, y_test = create_train_test_split(dataset)

    # create directories for the train and test images
    test_img_dir = output_filepath + "test"
    train_img_dir = output_filepath + "train"
    if not os.path.isdir(test_img_dir):
        os.mkdir(test_img_dir)
    if not os.path.isdir(train_img_dir):
        os.mkdir(train_img_dir)

    logger.info("save processed images")

    x_train_final = []
    x_test_final = []

    for idx in range(len(x_train)):
        img = Image.open(x_train[idx])
        img_name = train_img_dir + "/" + os.path.basename(x_train[idx])
        x_train_final.append(img_name)
        img.save(img_name)

    logger.info("Done with train images")

    for idx in range(len(x_test)):
        img = Image.open(x_test[idx])
        img_name = test_img_dir + "/" + os.path.basename(x_test[idx])
        x_test_final.append(img_name)
        img.save(img_name)

    logger.info("Done with test images")

    with open(output_filepath + "train_img_list", "wb") as fp:  # Pickling
        pickle.dump(x_train_final, fp)

    with open(output_filepath + "train_targets", "wb") as fp:  # Pickling
        pickle.dump(y_train, fp)

    with open(output_filepath + "test_img_list", "wb") as fp:  # Pickling
        pickle.dump(x_test_final, fp)

    with open(output_filepath + "test_targets", "wb") as fp:  # Pickling
        pickle.dump(y_test, fp)

    logger.info("Done making the final dataset")


def load_dataset(folder_path: str) -> dict:

    """
        Returns the dataset as a dict where the class names are also converted to english
        parameters:
            folder_path (str) the path to the raw data
        returns:
            dataset (dict) a dict containing the path to the images and the labels
    """

    categories = {
        "cane": "dog",
        "cavallo": "horse",
        "elefante": "elephant",
        "farfalla": "butterfly",
        "gallina": "chicken",
        "gatto": "cat",
        "mucca": "cow",
        "pecora": "sheep",
        "scoiattolo": "squirrel",
        "ragno": "spider",
    }
    dataset = []
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

    images_list = []
    labels_list = []
    for category, translate in categories.items():

        path = "data/raw/" + category
        target = animals.index(translate)

        for img in os.listdir(path):
            images_list.append(os.path.join(path, img))
            labels_list.append(target)

    dataset = {
        "images": images_list,
        "labels": labels_list,
    }

    return dataset


def create_train_test_split(dataset: dict) -> list:
    """
    spilt the data into train and test by stratifying it

    parameters:
        dataset (dict) containg the path to the images and the corresponding labels

    returns:
        x_train (list) containing the path to the train images
        x_test (list) containing the path to the test images
        y_train (list) containing the train labels
        y_test (list) containing the test labels

    """
    x = dataset["images"]
    y = dataset["labels"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
