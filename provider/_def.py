from argparse import Namespace
import cv2
import pickle as pkl
import numpy as np
import random
import torch
from torch.autograd import Variable

# model
from yolo.darknet import Darknet
from yolo.util import load_classes, write_results, write_to_image
from yolo.preprocess import prep_image

# settings
from settings import (yolo_weight,
                      yolo_conf,
                      yolo_classes,
                      yolo_color,
                      has_cuda)


def yolo_demo_detection(arguments: Namespace):
    # load network
    print("Running demo on YOLO")
    print("[YOLO] loading the network")
    model = Darknet(str(yolo_conf))
    model.load_weights(str(yolo_weight))
    model.eval()
    if has_cuda:
        model.cuda()
    print("[YOLO] Network had been loaded")
    model.net_info["height"] = arguments.yolo_dim
    inp_dim = int(model.net_info["height"])

    # load classes
    classes = load_classes(namesfile=str(yolo_classes))
    n_classes = len(classes)

    colors = pkl.load(open(str(yolo_color), "rb"))
    color = random.choice(colors)

    bbox_attrs = 5 + n_classes

    source = cv2.VideoCapture(arguments.in_file)

    while source.isOpened():
        ret, frame = source.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        if has_cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img), has_cuda)

        output = write_results(output, arguments.yolo_conf, n_classes, nms=True, nms_conf=arguments.yolo_nms_thresh)
        im_dim = im_dim.repeat(output.size(0), 1)

        scaling_factor = torch.min(inp_dim / im_dim, 1)[0].view(-1, 1)

        output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim[:, 0].view(-1, 1)) / 2
        output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim[:, 1].view(-1, 1)) / 2

        output[:, 1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim[i, 0])
            output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim[i, 1])

        list(map(lambda x: write_to_image(x, orig_im, classes, color), output))

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Main", frame)
        cv2.imshow("Det", orig_im)
    source.release()
    cv2.destroyAllWindows()
