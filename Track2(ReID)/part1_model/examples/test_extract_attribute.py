from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import math
import numpy as np
import sys
import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from reid.loss import TripletLoss

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.trainers import Cross_Trihard_Trainer
from reid.extract_attribute import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Attribute_Preprocessor, Flip_Preprocessor
from reid.utils.data.sampler import RandomIdentityAttributeSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, height, width, batch_size, num_instances,
        workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id, download=False)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.RectScale(height, width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(height, width),
        T.ToTensor(),
        normalizer,
    ])

    train_loader = DataLoader(
        Attribute_Preprocessor(train_set, root=dataset.images_dir,
                     transform=train_transformer),
        batch_size=batch_size, num_workers=workers,
        sampler=RandomIdentityAttributeSampler(train_set, num_instances),
        pin_memory=True, drop_last=True)

    test_loader = DataLoader(
        Attribute_Preprocessor(list(set(dataset.query) | set(dataset.gallery)),
                     root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, test_loader


def get_real_test_data(query_dir, gallery_dir, target_height, target_width, batch_size, workers):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.RectScale(target_height, target_width),
        T.ToTensor(),
        normalizer,
    ])  
    
    query_loader = DataLoader(
        Flip_Preprocessor(data_dir=query_dir, is_flip=False,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False,pin_memory=True)
    
    gallery_loader = DataLoader(
        Flip_Preprocessor(data_dir=gallery_dir, is_flip=False,
                     transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False,pin_memory=True)

    return query_loader, gallery_loader


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    #if args.height is None or args.width is None:
    #    args.height, args.width = (224, 224)
    #dataset, num_classes, train_loader, test_loader = \
    #    get_data(args.dataset, args.split, args.data_dir, args.height,
    #             args.width, args.batch_size, args.num_instances, args.workers,
    #             args.combine_trainval)

    query_loader, gallery_loader = get_real_test_data(args.query_dir, args.gallery_dir, args.height, args.width, args.batch_size, args.workers)
    #query_loader, _ = get_real_test_data(args.query_dir, args.gallery_dir, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = models.create(args.arch, num_classes=args.num_classes, num_features=args.features, test_attribute=True)

    # Load from checkpoint
    start_epoch = best_top1 = 0
    if args.weights: 
        checkpoint = load_checkpoint(args.weights)
        model.load_state_dict(checkpoint['state_dict'])
    if args.resume and not args.weights:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    # Test
    evaluator.evaluate(query_loader, gallery_loader)
    #evaluator.evaluate(query_loader, None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cross_entropy_trihard_net")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('--query_dir', type=str, default='./examples/data/image_query')
    parser.add_argument('--gallery_dir', type=str, default='./examples/data/image_test')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=224,
                        help="input height, default: 224")
    parser.add_argument('--width', type=int, default=224,
                        help="input width, default: 224")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--step_size', type=int, default=90)
    # model
    parser.add_argument('-a', '--arch', type=str, default='cross_entropy_trihard_resnet50',
                        choices=models.names())
    parser.add_argument('--num_classes', type=int, default=333)
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--lr_mult', type=float, default=1.0)
    parser.add_argument('--metric_loss_weight', type=float, default=0.02)
    parser.add_argument('--warm_up_ep', type=int, default=200)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--weights', type=str, default='', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--start_save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # metric learning
    parser.add_argument('--dist-metric', type=str, default='euclidean',
                        choices=['euclidean', 'kissme'])
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main(parser.parse_args())
