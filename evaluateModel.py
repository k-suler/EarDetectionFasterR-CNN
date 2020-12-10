import pdb
import cv2 as cv
from pprint import pprint
import numpy as np
import torch
from PIL import Image

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import itertools
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import json
from earDataset import EarDataset

CLASS_NAMES = ["__background__", "ear"]


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def get_prediction(model, img, confidence):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.

    """
    pred = model([img])
    import pprint

    # pprint.pprint(pred)
    pred_class = [CLASS_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [
        [(i[0], i[1]), (i[2], i[3])]
        for i in list(pred[0]['boxes'].detach().cpu().numpy())
    ]
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())

    pred_t = [pred_score.index(x) for x in pred_score if x > confidence][-1]

    pred_boxes = pred_boxes[: pred_t + 1]
    pred_class = pred_class[: pred_t + 1]
    pred_score = pred_score[: pred_t + 1]
    return pred_boxes, pred_class, pred_score


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
    # evaluate(model, data_loader_test, device=device)
    x = []
    ro = []
    for img, target in dataset_test:
        confidence = 0

        boxes, pred_cls, pred_score = get_prediction(model, img.to(device), confidence)

        tt = {}
        for i, item in enumerate(list(zip(boxes, pred_score))):
            pBox, score = item
            pBox = list(itertools.chain.from_iterable(pBox))
            for tBox in target['boxes'].tolist():
                iou = bb_intersection_over_union(pBox, tBox)
                if tt.get(i) is None or tt[i]['iou'] < iou:
                    tt[i] = {'iou': float(iou), 'score': float(score)}
        ro.extend(tt.values())
        x.append(
            {
                'img': target['image_id'].tolist()[0],
                'gt': target['boxes'].tolist(),
                'ious': tt,
            }
        )
        with open('gt/{}.txt'.format(target['image_id'].tolist()[0] + 1), 'w') as f:
            for box in target['boxes'].tolist():
                f.write('ear {}\n'.format(" ".join(map(str, box))))

        with open('dt/{}.txt'.format(target['image_id'].tolist()[0] + 1), 'w') as f:
            for box, score in zip(boxes, pred_score):

                box = list(itertools.chain.from_iterable(box))
                f.write('ear {} {}\n'.format(score, " ".join(map(str, box))))

    with open('iou.json', 'w') as f:
        f.write(json.dumps(ro))


if __name__ == "__main__":
    main()
