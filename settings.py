import os
from pathlib import Path

import torch

project_dir = Path(os.path.dirname(os.path.abspath(__file__)))

has_cuda = torch.cuda.is_available()

yolo_weight = project_dir.joinpath("data/yolo/yolov3.weights")

yolo_conf = project_dir.joinpath("data/yolo/cfg/yolov3.cfg")

yolo_classes = project_dir.joinpath("data/yolo/coco.names")

yolo_color = project_dir.joinpath("data/yolo/pallete")

yolo_filter_list = ["car", "motorbike", "truck"]
