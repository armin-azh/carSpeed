from argparse import Namespace
import cv2
import pickle as pkl
import numpy as np
import random
import torch
from torch.autograd import Variable
from torchvision import transforms

import matplotlib as mpl
import matplotlib.cm as cm

# yolo model
from yolo.darknet import Darknet
from yolo.util import load_classes, write_results, write_to_image, filter_list
from yolo.preprocess import prep_image

# mono-depth model
from mono.networks import ResnetEncoder, DepthDecoder
from mono.layers import disp_to_depth

# settings
from settings import (yolo_weight,
                      yolo_conf,
                      yolo_classes,
                      yolo_color,
                      has_cuda,
                      mono_weight_dir)


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

        output = filter_list(output, classes)

        list(map(lambda x: write_to_image(x, orig_im, classes, color), output))

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Main", frame)
        cv2.imshow("Det", orig_im)
    source.release()
    cv2.destroyAllWindows()


def mono_demo_detection(arguments: Namespace):
    device = torch.device("cuda") if has_cuda else torch.device("cpu")

    encoder_path = mono_weight_dir.joinpath("encoder.pth")
    depth_decoder_path = mono_weight_dir.joinpath("depth.pth")

    source = cv2.VideoCapture(arguments.in_file)

    # load models
    print("Running demo on Mono-depth")
    print("[MONO] loading the network")
    encoder = ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    depth_decoder = DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()
    print("[MONO] Network had been loaded")

    while source.isOpened():
        ret, frame = source.read()

        if not ret:
            break
        frame = cv2.resize(frame, (640, 192), interpolation=cv2.INTER_LANCZOS4)
        origin_frame = frame.copy()
        origin_frame = cv2.cvtColor(origin_frame, cv2.COLOR_BGR2RGB)
        frame_shape = origin_frame.shape
        original_height, original_width = frame_shape[0], frame_shape[1]

        with torch.no_grad():
            input_image = transforms.ToTensor()(origin_frame).unsqueeze(0)
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            color_mapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            color_mapped_im = cv2.cvtColor(color_mapped_im, cv2.COLOR_RGB2BGR)

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Main", frame)
        cv2.imshow("Disparity", color_mapped_im)
    source.release()
    cv2.destroyAllWindows()
