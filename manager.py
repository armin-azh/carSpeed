import os
import argparse
from argparse import Namespace

# providers
from provider import yolo_demo_detection, mono_demo_detection, py_d_net_detection


def main(arguments: Namespace):
    if arguments.yolo_demo:
        yolo_demo_detection(arguments=arguments)

    elif arguments.mono_demo:
        mono_demo_detection(arguments=arguments)

    elif arguments.pyd_net_demo:
        py_d_net_detection(arguments=arguments)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # default options
    parser.add_argument("--in_file", help="Input video file", default="", type=str)
    # yolo options
    parser.add_argument("--yolo_demo", help="run yolo demo detection", action="store_true", )
    parser.add_argument("--yolo_dim", help="yolo_dimension", default=416, type=int)
    parser.add_argument("--yolo_conf", help="Object Confidence to filter predictions", default=0.5, type=float)
    parser.add_argument("--yolo_nms_thresh", help="NMS threshold", default=0.4, type=float)
    # mono depth options
    parser.add_argument("--mono_demo", help="run mono depth demo", action="store_true")
    # PyD-net options
    parser.add_argument("--pyd_net_demo", help="run pyd_net demo", action="store_true")
    args = parser.parse_args()

    main(arguments=args)
