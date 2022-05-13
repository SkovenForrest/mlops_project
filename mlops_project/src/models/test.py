
from model import  MyAwesomeModel

import torch
import hydra
from omegaconf import OmegaConf
import logging
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl




class AnimalDataset(Dataset):
    def __init__(self, labels, images):
        self.img_labels = labels
        self.img = images

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image =self.img[idx]
        print("img before",image.shape)
        image = transform_images(image)
        
        print("img after",image.shape)
        label = self.img_labels[idx]   
        return image, label


def transform_images(data):
    transform = transforms.Compose([transforms.Resize((256,256))])
    return transform(data)

#parser = argparse.ArgumentParser(description='Training arguments')
#parser.add_argument('--lr', default=0.1)
# add any additional argument that you want
#args = parser.parse_args(sys.argv[2:])
#print(args)
dir = "C:\\Users\\Tobias\\Documents\\DTU\mlops_project\\mlops_project\\data\\processed\\"
#train_images = torch.load(dir + "train_images.pt")
#train_labels = torch.load(dir + "train_labels.pt")
test_images = torch.load(dir + "test_images.pt")
test_labels = torch.load(dir + "test_labels.pt")

print(test_labels)

import cv2
import os
categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel","ragno":"spider"}
dataset = []
animals = ["dog", "horse","elephant", "butterfly",  "chicken",  "cat", "cow",  "sheep", "squirrel", "spider"]


images = []
labels = []
for category,translate in categories.items():

    path = "data/raw/" + category
    #path = folder_path + category
    print("translate",translate)
    target = animals.index(translate)
    print("target", target)
    print(" ")
    
    for img in os.listdir(path):
        img=cv2.cvtColor(cv2.imread(os.path.join(path,img)), cv2.COLOR_BGR2RGB)
        #dataset.append([img,target])
        #print(dataset["target"])
    
        images.append(img)
        labels.append(target)

print(labels)

dataset = {
    'images': images,
    'label': labels,
}

    
#train_dataset = AnimalDataset(train_labels ,train_images)
#test_dataset  = AnimalDataset(test_labels,test_images)