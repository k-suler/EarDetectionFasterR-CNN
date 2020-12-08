import pdb
import cv2 as cv

import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T
import json
from earDataset import EarDataset


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load("./output/faster-rcnn-ear.pt")
    # our dataset has two classes only - background and ear
    num_classes = 2

    dataset_test = EarDataset(
        'AWEForSegmentation', 'test', 'testannot', get_transform(train=False)
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )

    # get the model using our helper function
    # move model to the right device
    model.to(device)
    evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    main()
