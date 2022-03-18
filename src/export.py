from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']


def export(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    opt.debug = max(opt.debug, 1)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    detector.export(32, 3, 384, 384, opt.output_file, opset_ver=10)


if __name__ == '__main__':
    opt = opts().init()
    export(opt)
