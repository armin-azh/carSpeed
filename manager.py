import os
import argparse
from argparse import Namespace

# providers
from provider import yolo_demo_detection


def main(arguments: Namespace):
    if arguments.yolo_demo:
        yolo_demo_detection(arguments=arguments)


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
    # PyD-net options
    args = parser.parse_args()

    main(arguments=args)
