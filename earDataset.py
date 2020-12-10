import os
import cv2 as cv

import torch
from PIL import Image

import transforms as T


class EarDataset(object):
    def __init__(self, root, images_path, masks_path, transforms):
        self.root = root
        self.transforms = transforms
        self.images_path = images_path
        self.masks_path = masks_path
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, images_path))))
        self.masks = list(sorted(os.listdir(os.path.join(root, masks_path))))

    def thresh_callback(self, val, src_gray):
        def transform_box(x, y, x1, y1):
            return (x, y, x + x1, y + y1)

        threshold = val

        canny_output = cv.Canny(src_gray, threshold, threshold * 2)
        contours, _ = cv.findContours(
            canny_output, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
        )
        contours_poly = [None] * len(contours)
        boundRect = [None] * len(contours)
        centers = [None] * len(contours)
        radius = [None] * len(contours)

        for i, c in enumerate(contours):
            contours_poly[i] = cv.approxPolyDP(c, 1, True)
            boundRect[i] = cv.boundingRect(contours_poly[i])
            if boundRect[i]:
                boundRect[i] = transform_box(*boundRect[i])
        return boundRect

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, self.images_path, self.imgs[idx])
        mask_path = os.path.join(self.root, self.masks_path, self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        mask = cv.imread(cv.samples.findFile(mask_path))
        b, g, r = cv.split(mask)
        r = r * 255
        g = g * 255
        b = b * 255
        mask = cv.merge((r, g, b))
        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
        mask = cv.blur(mask, (3, 3))

        thresh = 255  # initial threshold
        boxes = self.thresh_callback(thresh, mask)

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        num_objs = len(boxes)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        # masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        # target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
