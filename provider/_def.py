import warnings

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

from argparse import Namespace
import cv2
import pickle as pkl
import time
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

# py-d net model
from pydnet import PydNet

# settings
from settings import (yolo_weight,
                      yolo_conf,
                      yolo_classes,
                      yolo_color,
                      has_cuda,
                      mono_weight_dir,
                      tracker_iou_threshold,
                      tracker_min_hits,
                      tracker_max_age,
                      tracker_interval,
                      video_size)

# tracker
from tracklet import SortTrackerV1
from tracklet import Counter
from tracklet import TrackerContainer


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


def py_d_net_detection(arguments: Namespace):
    device = torch.device("cuda") if has_cuda else torch.device("cpu")

    source = cv2.VideoCapture(arguments.in_file)

    # load model
    print("Running demo on PydNet-depth")
    print("[PydNet] loading the network")
    cont = PydNet()
    model = cont.model
    model.to(device)
    transformer = cont.transformer.small_transform
    print("[YOLO] Network had been loaded")

    while source.isOpened():
        ret, frame = source.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = cv2.resize(frame, (640, 192), interpolation=cv2.INTER_LANCZOS4)

        trans_frame = transformer(frame).to(device)

        with torch.no_grad():
            pred = model(trans_frame)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            output = pred.cpu().numpy()

            vmax = np.percentile(output, 95)
            normalizer = mpl.colors.Normalize(vmin=output.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            color_mapped_im = (mapper.to_rgba(output)[:, :, :3] * 255).astype(np.uint8)
            color_mapped_im = cv2.cvtColor(color_mapped_im, cv2.COLOR_RGB2BGR)

        if cv2.waitKey(1) == ord("q"):
            break

        cv2.imshow("Main", frame)
        cv2.imshow("Disparity", color_mapped_im)
    source.release()
    cv2.destroyAllWindows()


def yolo_mono(arguments: Namespace) -> None:
    """
    start yolo + MonoDepth runtime
    :param arguments: runtime argument
    :return:
    """
    # use gpu for boosting if you have
    device = torch.device("cuda") if has_cuda else torch.device("cpu")

    # mono-depth paths
    encoder_path = mono_weight_dir.joinpath("encoder.pth")
    depth_decoder_path = mono_weight_dir.joinpath("depth.pth")

    # load network
    # load yolo
    print("Running demo on YOLO + MonoDepth2")
    print("[YOLO] loading the network")
    model = Darknet(str(yolo_conf))
    model.load_weights(str(yolo_weight))
    model.eval()
    if has_cuda:
        model.cuda()
    print("[YOLO] Network had been loaded")
    model.net_info["height"] = arguments.yolo_dim
    inp_dim = int(model.net_info["height"])
    # end load yolo
    # start load mono
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
    # end load mono

    # start load classes
    classes = load_classes(namesfile=str(yolo_classes))
    n_classes = len(classes)
    # end load classes

    # start load video source
    source = cv2.VideoCapture(arguments.in_file if not arguments.cam else 0)
    # end load video source

    colors = pkl.load(open(str(yolo_color), "rb"))
    color = random.choice(colors)

    # tracker
    tracker = SortTrackerV1(iou_threshold=tracker_iou_threshold,
                            max_age=tracker_max_age,
                            min_hit=tracker_min_hits,
                            detect_interval=tracker_interval)
    interval_cnt = Counter()

    trk_store = dict()

    o_height, o_width = video_size
    cp_frame = source.get(cv2.CAP_PROP_FPS)

    print(f"[Video] width: {o_width}, height: {o_height}, FPS: {round(cp_frame,2)}")

    # start detection
    while source.isOpened():
        ret, frame = source.read()
        start_time = time.time()

        if not ret:
            break

        origin_frame = cv2.resize(frame, (o_width, o_height))

        frame = cv2.resize(frame, (640, 480))
        frame_shape = frame.shape
        frame_height, frame_width = frame_shape[0], frame_shape[1]

        # start some transfer
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)

        # start calc scale
        o_f_h_scale = o_height / orig_im.shape[0]
        o_f_w_scale = o_width / orig_im.shape[1]
        # end calc scale

        if has_cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()
        # end some transfer

        # start mono conversion
        with torch.no_grad():
            input_image = transforms.ToTensor()(frame).unsqueeze(0)
            input_image = input_image.to(device)
            features = encoder(input_image)
            mono_output = depth_decoder(features)

            disp = mono_output[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(disp, (frame_height, frame_width), mode="bilinear",
                                                           align_corners=False)

            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)

            mono_depth = depth.cpu().numpy()

            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            color_mapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            color_mapped_im = cv2.cvtColor(color_mapped_im, cv2.COLOR_RGB2BGR)
            color_mapped_im = cv2.resize(color_mapped_im, (o_width, o_height))

        if interval_cnt() % tracker_interval == 0:
            interval_cnt.reset()
            # start yolo prediction
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
            output = np.array(output)
            # end yolo prediction
        else:
            output = []

        interval_cnt.next()

        trk_output = tracker.detect(road_objets=output, frame=orig_im, frame_size=orig_im.shape[:2])

        mono_depth = np.squeeze(np.squeeze(mono_depth, axis=0), axis=0)

        for box in trk_output:
            box = box.astype(np.int)
            x1, y1, x2, y2 = box[:4]

            # start calc depth
            car_depth = mono_depth[y1:y2, x1:x2]
            car_depth_mean = np.mean(car_depth)
            # end calc depth

            obj_idx = box[4]
            obj = trk_store.get(str(obj_idx))
            obj_speed = None
            if obj is None:
                trk_store[str(obj_idx)] = TrackerContainer()
                trk_store[str(obj_idx)].get_speed(mean_depth=car_depth_mean)
            else:
                obj_speed = trk_store[str(obj_idx)].get_speed(mean_depth=car_depth_mean)
                obj_speed = round(arguments.ref_speed + obj_speed, 2)

            label = f"ID: {obj_idx}"
            speed = f"Speed: {obj_speed}"

            # start transformation
            x1 = int(x1 * o_f_w_scale)
            y1 = int(y1 * o_f_h_scale)
            x2 = int(x2 * o_f_w_scale)
            y2 = int(y2 * o_f_h_scale)
            # end transformation

            # start annotating information
            cv2.rectangle(origin_frame, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(origin_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], 1)
            cv2.putText(origin_frame, speed, (x1, y1 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
            # start annotate on real image

            # start annotate on depth image
            cv2.rectangle(color_mapped_im, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(color_mapped_im, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], 1)
            cv2.putText(color_mapped_im, speed, (x1, y1 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
            # end annotating information

        if cv2.waitKey(1) == ord("q"):
            break

        # start show frame rate
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(origin_frame, f"FPS: {round(fps, 2)}", (12, 15), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 2)
        # end show frame rate

        # start window
        cat_frame = np.concatenate([color_mapped_im, origin_frame], axis=1)
        cv2.imshow("Yolo + Mono", cat_frame)
        # end window

    source.release()
    cv2.destroyAllWindows()


def yolo_pyd_net(arguments: Namespace) -> None:
    """
    start yolo + PyDNet runtime
    :param arguments:
    :return: None
    """
    # use gpu for boosting if you have
    device = torch.device("cuda") if has_cuda else torch.device("cpu")

    # load network
    # load yolo
    print("Running demo on YOLO + PyDNet")
    print("[YOLO] loading the network")
    model = Darknet(str(yolo_conf))
    model.load_weights(str(yolo_weight))
    model.eval()
    if has_cuda:
        model.cuda()
    print("[YOLO] Network had been loaded")
    model.net_info["height"] = arguments.yolo_dim
    inp_dim = int(model.net_info["height"])
    # end load yolo
    # start load mono
    print("[PydNet] loading the network")
    cont = PydNet()
    py_model = cont.model
    py_model.to(device)
    transformer = cont.transformer.small_transform
    print("[PydNet] Network had been loaded")
    # end load mono

    # start load classes
    classes = load_classes(namesfile=str(yolo_classes))
    n_classes = len(classes)
    # end load classes

    # start load video source
    source = cv2.VideoCapture(arguments.in_file if not arguments.cam else 0)
    # end load video source

    colors = pkl.load(open(str(yolo_color), "rb"))
    color = random.choice(colors)

    # tracker
    tracker = SortTrackerV1(iou_threshold=tracker_iou_threshold,
                            max_age=tracker_max_age,
                            min_hit=tracker_min_hits,
                            detect_interval=tracker_interval)
    interval_cnt = Counter()

    trk_store = dict()

    o_height, o_width = video_size
    cp_frame = source.get(cv2.CAP_PROP_FPS)

    print(f"[Video] width: {o_width}, height: {o_height}, FPS: {round(cp_frame,2)}")

    # start detection
    while source.isOpened():
        ret, frame = source.read()
        start_time = time.time()

        if not ret:
            break

        origin_frame = cv2.resize(frame, (o_width, o_height))

        # start some transformation
        frame = cv2.resize(frame, (640, 192))
        img, orig_im, dim = prep_image(frame, inp_dim)
        im_dim = torch.FloatTensor(dim).repeat(1, 2)
        # end some transformation

        # start calc scale
        o_f_h_scale = o_height / orig_im.shape[0]
        o_f_w_scale = o_width / orig_im.shape[1]
        # end calc scale

        # start moving to gpu
        if has_cuda:
            im_dim = im_dim.cuda()
            img = img.cuda()
        # end moving to gpu

        # start pyd-net detection
        trans_frame = transformer(frame).to(device)
        with torch.no_grad():
            py_pred = py_model(trans_frame)
            # # start py-d net postprocessing
            py_pred = torch.nn.functional.interpolate(
                py_pred.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

            py_depth = py_pred.cpu().numpy()

            vmax = np.percentile(py_depth, 95)
            normalizer = mpl.colors.Normalize(vmin=py_depth.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            color_mapped_im = (mapper.to_rgba(py_depth)[:, :, :3] * 255).astype(np.uint8)
            color_mapped_im = cv2.cvtColor(color_mapped_im, cv2.COLOR_RGB2BGR)
            color_mapped_im = cv2.resize(color_mapped_im, (o_width, o_height))
            # end yolo postprocessing
        # end pyd-net detection

        # check the interval
        if interval_cnt() % tracker_interval == 0:
            interval_cnt.reset()
            # start yolo prediction
            with torch.no_grad():
                output = model(Variable(img), has_cuda)

            # start yolo postprocessing
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
            output = np.array(output)
            # end yolo postprocessing
            # end yolo prediction
        else:
            output = []

        interval_cnt.next()

        trk_output = tracker.detect(road_objets=output, frame=orig_im, frame_size=orig_im.shape[:2])

        py_depth = 1 / py_depth

        # start representing
        for box in trk_output:
            box = box.astype(np.int)
            x1, y1, x2, y2 = box[:4]
            car_depth = py_depth[y1:y2, x1:x2]
            car_depth_mean = np.mean(car_depth)
            obj_idx = box[4]
            obj = trk_store.get(str(obj_idx))
            obj_speed = None

            # start track the object
            if obj is None:
                trk_store[str(obj_idx)] = TrackerContainer()
                trk_store[str(obj_idx)].get_speed(mean_depth=car_depth_mean)
            else:
                obj_speed = trk_store[str(obj_idx)].get_speed(mean_depth=car_depth_mean)
                obj_speed = round(arguments.ref_speed + obj_speed, 2)
            # end track the object

            label = f"ID: {obj_idx}"
            speed = f"Speed: {obj_speed}"

            # start transformation
            x1 = int(x1 * o_f_w_scale)
            y1 = int(y1 * o_f_h_scale)
            x2 = int(x2 * o_f_w_scale)
            y2 = int(y2 * o_f_h_scale)
            # end transformation

            # start annotate on real image
            cv2.rectangle(origin_frame, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(origin_frame, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], 1)
            cv2.putText(origin_frame, speed, (x1, y1 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
            # start annotate on real image

            # start annotate on depth image
            cv2.rectangle(color_mapped_im, (x1, y1), (x2, y2), color, 1)
            # cv2.putText(color_mapped_im, label, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [225, 255, 255], 1)
            cv2.putText(color_mapped_im, speed, (x1, y1 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [255, 255, 255], 1)
            # end annotate on depth image
        #  end representing

        # start some wait
        if cv2.waitKey(1) == ord("q"):
            break
        # end some wait

        # start show frame rate
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(origin_frame, f"FPS: {round(fps,2)}", (12, 15), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 2)
        # end show frame rate

        cat_frame = np.concatenate([color_mapped_im, origin_frame], axis=1)
        # start opening window
        cv2.imshow("YOLO + PyDNet", cat_frame)
        # end opening window

    source.release()
    cv2.destroyAllWindows()
