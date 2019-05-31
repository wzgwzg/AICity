from __future__ import print_function, absolute_import

import argparse
import sys
import os.path as osp
from collections import defaultdict
import math
import numpy as np
from PIL import Image
import cPickle
import pickle

import torch
from torch import nn
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.evaluators import extract_features
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Preprocessor, PreprocessorWithMasks
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    model_type = args.model_type
    # params
    params = {}
    if model_type == 'masks':
        params['num_m_features'] = args.num_m_features
        params['masks'] = args.masks
    else:
        print('unkrown model type.')
        return

    # Create model
    model = models.create(args.arch, num_classes=333, params=params)
    
    checkpoint = load_checkpoint(args.weights)
    model.load_state_dict(checkpoint['state_dict'])

    model = nn.DataParallel(model, [0,1,2,3]).cuda()
    model.eval()

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    transform = T.Compose([
        T.ToTensor(),
        normalize,
    ])

    image_batch = []; masks_batch = []; image_name_batch = []
    batch_size = args.batch_size
    feature_dict = {}
    with open(args.input_list, 'r') as fp:
        for i, line in enumerate(fp):
            image_name = line.strip()
            image_name_batch.append(image_name)
            image_file = osp.join(args.image_dir, image_name)
            image = Image.open(image_file).convert('RGB')
            image = image.resize((args.width, args.height), Image.BILINEAR)
            image = transform(image)
            image_batch.append(image.view(1, image.size(0), image.size(1), image.size(2)))
            if model_type == 'masks':
                masks_file = osp.join(args.masks_dir, image_name+'.pkl')
                with open(masks_file, 'rb') as f:
                    masks = cPickle.load(f)
                masks = torch.from_numpy(masks).type_as(image)
                masks_batch.append(masks.view(1, masks.size(0), masks.size(1), masks.size(2)))
        
            if (i+1) % batch_size == 0:
                image_batch = Variable(torch.cat(image_batch, 0)).cuda()
                if model_type == 'masks':
                    masks_batch = Variable(torch.cat(masks_batch, 0)).cuda()
                    inputs = [image_batch, masks_batch]
                else:
                    inputs = [image_batch, None]

                outputs = model(*inputs)
                features = outputs.view(batch_size, -1).data.cpu().numpy()
                for b in range(batch_size):
                    feature_dict[image_name_batch[b]] = features[b]

                image_name_batch = []
                image_batch = []
                masks_batch = []

            if i % 100 == 0:
                print('%d images processed.' % (i+1))

        cnt = len(image_name_batch)
        if cnt > 0:
            image_batch = Variable(torch.cat(image_batch, 0)).cuda()
            if model_type == 'masks':
                masks_batch = Variable(torch.cat(masks_batch, 0)).cuda()
                inputs = [image_batch, masks_batch]
            else:
                inputs = [image_batch, None]
            
            outputs = model(*inputs)
            features = outputs.view(cnt, -1).data.cpu().numpy()
            for b in range(cnt):
                feature_dict[image_name_batch[b]] = features[b]

            image_name_batch = []
            image_batch = []
            masks_batch = []

    with open(osp.join(args.output_dir, args.output_pkl), 'wb') as fp:
        pickle.dump(feature_dict, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="init parameters")
    # io
    parser.add_argument('--input-list', required=True, type=str)
    parser.add_argument('--image-dir', required=True, type=str)
    parser.add_argument('--masks-dir', required=True, type=str)
    parser.add_argument('--output-dir', required=True, type=str)
    parser.add_argument('--output-pkl', required=True, type=str)
    # data
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=256,
                        help="input width, default: 256")
    # model
    parser.add_argument('-a', '--arch', type=str, default='',
                        choices=models.names())
    parser.add_argument('--weights', type=str, default='', metavar='PATH')
    parser.add_argument('--model-type', type=str, default='global')
    parser.add_argument('--num-m-features', type=int, default=256)
    parser.add_argument('--masks', type=int, default=6)
    # configs
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
     
    main(parser.parse_args())
