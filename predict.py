import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
import os
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument(
    "-m", "--model", default="./output/faster-rcnn-ear.pt", help="path to the model"
)
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-a", "--all", default=False)
ap.add_argument("-s", "--save", default=False)
ap.add_argument(
    "-c", "--confidence", type=float, default=0.7, help="confidence to keep predictions"
)
args = vars(ap.parse_args())

CLASS_NAMES = ["__background__", "ear"]


def get_prediction(img_path, confidence):
    img = Image.open(img_path)
    transform = T.Compose([T.ToTensor()])
    img = transform(img).to(device)

    pred = model([img])
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


def detect_object(
    img_path, confidence=0.5, rect_th=2, text_size=0.5, text_th=1, save=False
):
    try:
        boxes, pred_cls, pred_score = get_prediction(img_path, confidence)
    except:
        return None
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(len(boxes))
    for i in range(len(boxes)):
        cv2.rectangle(
            img, boxes[i][0], boxes[i][1], color=(0, 255, 0), thickness=rect_th
        )
        cv2.putText(
            img,
            pred_cls[i] + ": " + str(round(pred_score[i], 3)),
            boxes[i][0],
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            (0, 255, 0),
            thickness=text_th,
        )
    fig = plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    if args['save']:

        spl = imgP.split('/')
        plt.savefig("./output/predictions/{}".format(spl[-1]), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show(block=True)


if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = torch.load(args["model"])
    img_path = args["image"]

    if args['all']:
        root = "./AWEForSegmentation/test/"
        imgs = list(sorted(os.listdir(os.path.join('./AWEForSegmentation/test/'))))
        for imgP in imgs:
            print(root + imgP)
            detect_object(root + imgP, confidence=args["confidence"])
    else:
        detect_object(img_path, confidence=args["confidence"])
