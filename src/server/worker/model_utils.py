import logging
import os
import time
import torch
import torch.backends.cudnn as cudnn
from .tool.darknet2pytorch import Darknet
from .tool.torch_utils import do_detect
import cv2
from src.utils import FRAME_SIZES, sleepInMillis
import numpy as np
import random


def model_predict(model, imgs, use_gpu=1, gpu_number=0):
    return do_detect(model, imgs, conf_thresh=0.20, nms_thresh=0.4,
                     use_cuda=use_gpu, gpu_number=gpu_number)


def set_cuda_device(gpu_number):
    torch.cuda.set_device(gpu_number)


def set_deterministic_behaviour(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.enabled = False
    cudnn.deterministic = True
    cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def model_input_resize(model_number, input_img):
    frame_size = FRAME_SIZES[model_number]
    if input_img.shape[0] != frame_size:
        logging.info(f"ModelInputResize {input_img.shape[0]}"
                     f" does not match with model size {frame_size}")
        img = cv2.resize(input_img, (frame_size, frame_size),
                         cv2.INTER_NEAREST)
    else:
        img = input_img

    return img


def unserialize(raw_bytes):
    img = cv2.imdecode(np.fromstring(
        raw_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def load_model(opts, frame_size):
    cfg_file_path = opts.model_config_dir + \
        "/yolov4_" + str(frame_size) + ".cfg"
    model = Darknet(cfg_file_path, inference=True)
    weight_file = os.path.join(
        opts.weights_dir, "yolov4_{}.pth".format(frame_size))
    checkpoint = torch.load(
        weight_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    model.eval()

    cudnn.benchmarks = True
    cudnn.enabled = True

    # Zero grad for parameters
    for param in model.parameters():
        param.grad = None
    return model


class SimulatedPytorchModel:
    def __init__(self):
        pass

    @staticmethod
    def cpu():
        pass

    @staticmethod
    def cuda(gpu_number):
        pass

    def share_memory(self):
        return self

    @staticmethod
    def predict(imgs, model_time, start_time):
        _OFFSET_OVERHEAD = 2
        output = np.empty((imgs.shape[0], 0, 7), dtype=np.float64)
        elapsed_time = (time.time() - start_time) * 1e3 + _OFFSET_OVERHEAD
        sleepInMillis((model_time-elapsed_time))
        return output


def generate_model(opts, model_number, use_gpu=None, gpu_number=0):
    if not opts.simulate_gpu:
        model = load_model(opts, FRAME_SIZES[model_number])
    else:
        model = SimulatedPytorchModel()
    if use_gpu:
        model.cuda(gpu_number)
    return model
