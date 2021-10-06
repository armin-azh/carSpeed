import os
from pathlib import Path

import torch

project_dir = Path(os.path.dirname(os.path.abspath(__file__)))

has_cuda = torch.cuda.is_available()

# yolo
yolo_weight = project_dir.joinpath("data/yolo/yolov3.weights")

yolo_conf = project_dir.joinpath("data/yolo/cfg/yolov3.cfg")

yolo_classes = project_dir.joinpath("data/yolo/coco.names")

yolo_color = project_dir.joinpath("data/yolo/pallete")

yolo_filter_list = ["car", "motorbike", "truck"]

# mono depth
mono_weight_dir = project_dir.joinpath("data/monodepth/mono_640x192")

# pyd-net depth
py_d_net = project_dir.joinpath("data/pydnet/pydnet")

# tracker setting
tracker_iou_threshold = 0.2
tracker_max_age = 2
tracker_min_hits = 12
tracker_interval = 6

# time step
look_back = 10

# video
video_size = (480, 854)  # 480p
# video_size = (720, 1280)  # 720p
# video_size = (360, 640)  # 360p

