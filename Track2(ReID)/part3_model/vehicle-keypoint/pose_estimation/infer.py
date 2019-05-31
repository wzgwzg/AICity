# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import cPickle
import numpy as np
import cv2

import torch
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable

import _init_paths
from core.config import config
from core.config import update_config
from core.inference import get_max_preds

import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # input 
    parser.add_argument('--input-dir',
                        help='input image dir',
                        required=True,
                        type=str)
    parser.add_argument('--input-list',
                        help='input image list',
                        required=True,
                        type=str)
    parser.add_argument('--model-file',
                        help='model state file',
                        required=True,
                        type=str)
    parser.add_argument('--output-dir',
                        help='output image dir',
                        required=True,
                        type=str)

    # settings
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--batch-size',
                        help='batchsize',
                        type=int)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file


def resize_heatmap(hms, height, width):
    num = hms.shape[0]
    resized_hms = np.zeros((num, height, width))
    for i in range(num):
        resized_hms[i] = cv2.resize(hms[i], (width, height))
    resized_hms = resized_hms.astype('float')
    return resized_hms


def visualize(config, image_names, images, heatmaps, output_dir):
    # get size
    batch_size, num_joints, heatmap_h, heatmap_w = heatmaps.shape

    # get keypoints 
    coords, maxvals = get_max_preds(heatmaps)
    
    # save heatmap
    for i in range(batch_size): 
        resized_heatmap = resize_heatmap(heatmaps[i], 18, 24)
        with open(os.path.join(output_dir, image_names[i]+'.pkl'), 'wb') as f:
            cPickle.dump(resized_heatmap, f, protocol=cPickle.HIGHEST_PROTOCOL)


def main():
    args = parse_args()
    reset_config(config, args)

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    # model
    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()
    model.load_state_dict(torch.load(config.TEST.MODEL_FILE))

    # preprocessing
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # inference 
    with open(args.input_list, 'r') as fp:
        total_cnt = 0
        cnt = 0
        image_names = []
        images = []
        inputs = np.zeros((args.batch_size, 3, \
                    config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]), dtype=np.float32)
        for line in fp:
            image_file = os.path.join(args.input_dir, line.strip())
            image_names.append(os.path.basename(image_file))
            image = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            images.append(image)
            image = cv2.resize(image, (config.MODEL.IMAGE_SIZE[0], config.MODEL.IMAGE_SIZE[1]))
            input = transform(image) 
            inputs[cnt, :, :, :] = input
            cnt += 1

            if cnt % args.batch_size == 0:
                # forward
                inputs = Variable(torch.from_numpy(inputs).float()).cuda()
                outputs = model(inputs)
                outputs = outputs.data.cpu().numpy()
                total_cnt += cnt
                print('%d images processed' % total_cnt)
                # visualize
                visualize(config, image_names, images, outputs, args.output_dir)
                # reset
                cnt = 0
                image_names = []
                images = []
                inputs = np.zeros((args.batch_size, 3, \
                            config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]), dtype=np.float32)

        if cnt > 0:
            # forward
            inputs = Variable(torch.from_numpy(inputs).float()).cuda()
            outputs = model(inputs)
            outputs = outputs.data.cpu().numpy()
            outputs = outputs[0:cnt, :, :, :]
            total_cnt += cnt
            print('%d images processed' % total_cnt)
            # visualize
            visualize(config, image_names, images, outputs, args.output_dir)
            # reset
            cnt = 0
            image_names = []
            images = []
            inputs = np.zeros((args.batch_size, 3, \
                        config.MODEL.IMAGE_SIZE[1], config.MODEL.IMAGE_SIZE[0]), dtype=np.float32)
            
            
if __name__ == '__main__':
    main()
