# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import kornia as K
import cv2
import glob
from torchvision import transforms
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
import random
from torch.utils.data import DataLoader, Dataset
import pickle

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # dataset from https://www.kaggle.com/datasets/alessiocorrado99/animals10

    #terminal command python src/data/make_dataset.py data/raw/ data/processed/  

    logger.info('loading the images')
    dataset = load_dataset(input_filepath)

    logger.info('making the train and test split')
    x_train, x_test, y_train, y_test = create_train_test_split(dataset)
    #std tensor([0.2765, 0.2734, 0.2861])
    #mean tensor([0.5320, 0.5095, 0.4346])

    test_img_dir = output_filepath+ "test"
    train_img_dir = output_filepath+ "train"
    if(not os.path.isdir(test_img_dir)):
        os.mkdir(test_img_dir)
    if(not os.path.isdir(train_img_dir)):    
        os.mkdir(train_img_dir)

    logger.info('save processed images')

    x_train_final = []
    x_test_final = []

    for idx in range(len(x_train)):
        img = cv2.cvtColor(cv2.imread(x_train[idx]), cv2.COLOR_BGR2RGB)
        img_name = train_img_dir + "/" + os.path.basename(x_train[idx])
        x_train_final.append(img_name)
        cv2.imwrite(img_name,img)
    
    logger.info('Done with train images')

    for idx in range(len(x_test)):
        img = cv2.cvtColor(cv2.imread(x_test[idx]), cv2.COLOR_BGR2RGB)
        img_name = test_img_dir + "/" + os.path.basename(x_test[idx])
        x_test_final.append(img_name)
        cv2.imwrite(img_name,img)

    logger.info('Done with test images')


    with open(output_filepath+"train_img_list", "wb") as fp:   #Pickling
        pickle.dump(x_train_final, fp)
    
    with open(output_filepath+"train_targets", "wb") as fp:   #Pickling
        pickle.dump(y_train, fp)

    with open(output_filepath+"test_img_list", "wb") as fp:   #Pickling
        pickle.dump(x_test_final, fp)
    
    with open(output_filepath+"test_targets", "wb") as fp:   #Pickling
        pickle.dump(y_test, fp)


    #with open("test", "rb") as fp:   # Unpickling
    #    b = pickle.load(fp)
    """
    logger.info('converting images to tensors and normalizing them')
    for idx in range(len(x_train)):
        x_train[idx] = transform_images(x_train[idx])

    for idx in range(len(x_test)):
        x_test[idx] = transform_images(x_test[idx])

    logger.info('saving the dataset')
    torch.save(x_train,output_filepath + "train_images.pt")
    torch.save(x_test,output_filepath + "test_images.pt")
    torch.save(y_train,output_filepath + "train_labels.pt")
    torch.save(y_test,output_filepath + "test_labels.pt")

    """
    
    logger.info('Done making the final dataset')

def load_dataset(folder_path):
    categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel","ragno":"spider"}
    dataset = []
    animals = ["dog", "horse","elephant", "butterfly",  "chicken",  "cat", "cow",  "sheep", "squirrel", "spider"]


    images_list = []
    labels_list = []
    for category,translate in categories.items():

        path = "data/raw/" + category
        target = animals.index(translate)
        
        for img in os.listdir(path):
            images_list.append(os.path.join(path,img))
            labels_list.append(target)

    dataset = {
        'images': images_list,
        'labels': labels_list,
    }

    return dataset

def create_train_test_split(dataset):  
    x = dataset["images"]
    y = dataset["labels"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify= y)
    return x_train, x_test, y_train, y_test

"""

def transform_images(data):
    transform = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),transforms.Normalize((0.5320, 0.5095, 0.4346), (0.2765, 0.2734, 0.2861))])
    return transform(data)

def load_dataset(folder_path):
    categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel","ragno":"spider"}
    dataset = []
    animals = ["dog", "horse","elephant", "butterfly",  "chicken",  "cat", "cow",  "sheep", "squirrel", "spider"]


    images = []
    labels = []
    for category,translate in categories.items():

        path = "data/raw/" + category
        target = animals.index(translate)
        
        for img in os.listdir(path):
            img=cv2.cvtColor(cv2.imread(os.path.join(path,img)), cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(target)

    dataset = {
        'images': images,
        'labels': labels,
    }
    return dataset

def create_train_test_split(dataset):  
    x = dataset["images"]
    y = dataset["labels"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify= y)
    return x_train, x_test, y_train, y_test

"""


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

