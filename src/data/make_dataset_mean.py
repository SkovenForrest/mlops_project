# -*- coding: utf-8 -*-
import os
import random

import cv2
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_dataset(folder_path):
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
    for category, translate in categories.items():

        path = "mlops_project/data/raw/" + category
        target = animals.index(translate)

        for img in os.listdir(path):
            img = cv2.cvtColor(cv2.imread(os.path.join(path, img)), cv2.COLOR_BGR2RGB)
            dataset.append([img, target])

        return dataset


def create_train_test_split(dataset):
    x = []
    y = []
    random.seed(30)
    random.shuffle(dataset)

    for images, labels in dataset:
        x.append(images)
        y.append(labels)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    return x_train, x_test, y_train, y_test


dataset = load_dataset("mlops_project/data/raw/")

x_train, x_test, y_train, y_test = create_train_test_split(dataset)


def transform_data(data):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((256, 256))]
    )
    return transform(data)


class CustomDataset(Dataset):
    def __init__(self, labels, images):
        self.labels = labels
        self.img = images

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = transform_data(self.img[idx])
        labels = self.labels[idx]
        return image, labels


train_dataset = CustomDataset(y_train, x_train)
train_set = DataLoader(train_dataset, batch_size=10, shuffle=True)


def get_mean_and_std(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    print("std", std)
    print("mean", mean)
    return mean, std


get_mean_and_std(train_set)
