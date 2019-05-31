from __future__ import print_function, absolute_import

import argparse
import sys
import os.path as osp
import random
import math
import numpy as np

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.loss import TripletLoss
from reid.trainers import Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import PreprocessorWithMasks
from reid.utils.data.sampler import RandomIdentitySampler, RandomIdentityAndCameraSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, logs_dir, model_type, params,
             height, width, crop_height, crop_width,
             batch_size, workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id, download=False)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    if model_type == 'masks':
        train_transformer = T.Compose([
            T.ToTensor(),
            normalizer,
        ])
        test_transformer = T.Compose([
            T.ToTensor(),
            normalizer,
        ])
        train_preprocessor = PreprocessorWithMasks(train_set, root=dataset.images_dir, masks_root=dataset.masks_dir, \
                                                   height=height, width=width, num_masks=params['masks'], \
                                                   transform=train_transformer, is_training=True)
        test_preprocessor = PreprocessorWithMasks(list(set(dataset.query) | set(dataset.gallery)), root=dataset.images_dir, masks_root=dataset.masks_dir, \
                                                  height=height, width=width, num_masks=params['masks'], \
                                                  transform=test_transformer, is_training=False)
    else:
        print('unkrown model type.')
        return
         
    random_train_loader = DataLoader(
        train_preprocessor,
        sampler=RandomIdentityAndCameraSampler(train_set, num_instances=4),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)

    test_loader = DataLoader(
        test_preprocessor,
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, random_train_loader, test_loader


def main(args):
    random.seed(args.seed)
    np.random.seed(1)
    torch.manual_seed(1)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # params
    params = {}
    if args.model_type == 'masks':
        params['num_m_features'] = args.num_m_features
        params['masks'] = args.masks
    else:
        print('unkrown model type.')
        return

    # Create data loaders
    if args.height is None or args.width is None:
        args.height, args.width = (256, 256)
    dataset, num_classes, random_train_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.logs_dir, args.model_type, params,
                 args.height, args.width, args.crop_height, args.crop_width, 
                 args.batch_size, args.workers, args.combine_trainval)

    # Create model
    model = models.create(args.arch, num_classes=num_classes, params=params)

    # Load from checkpoint
    start_epoch = best_top1 = best_mAP = 0
    if args.weights: 
        checkpoint = load_checkpoint(args.weights)
        model_dict = model.state_dict()
        checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        model_dict.update(checkpoint_load)
        model.load_state_dict(model_dict)
    if args.resume and not args.weights:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best_top1))
    model = nn.DataParallel(model, [0,1,2,3]).cuda()

    # Criterion
    criterion = TripletLoss().cuda()

    # Optimizer
    base_params = []
    for name, p in model.module.base.named_parameters():
        base_params.append(p)
    base_param_ids = set(map(id, model.module.base.parameters()))
    new_params = [p for p in model.parameters() if
                  id(p) not in base_param_ids]
    if args.model_type == 'masks':
        param_groups = [ 
            {'params': base_params, 'lr_mult': args.lrm},
            {'params': new_params, 'lr_mult': 1.0}]
    else:
        print('unkrown model type.')
        return

    optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=True)

    # Trainer
    trainer = Trainer(model, criterion, num_classes, args.logs_dir)

    # Evaluator
    evaluator = Evaluator(model)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size = args.step_size 
        if epoch < step_size:
            lr = args.lr
        elif epoch >= step_size and epoch < args.epochs:
            lr = args.lr * 0.1
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        return lr

    # Start training
    for epoch in range(start_epoch, args.epochs):
        lr = adjust_lr(epoch)

        if epoch < args.warm_up_ep:
            trainer.train(epoch, random_train_loader, optimizer, lr, True, args.warm_up_ep)
        else:
            trainer.train(epoch, random_train_loader, optimizer, lr, False, args.warm_up_ep)
        
        if epoch < args.start_save:
            continue

        if epoch % 10 == 9:
            print('Epoch: [%d]' % epoch)
            top1, mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.module.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model_best.pth.tar'))

            if epoch == args.epochs - 1:
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch + 1,
                    'best_mAP': mAP,
                }, True, fpath=osp.join(args.logs_dir, 'last.pth.tar')) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="reid")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=2)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--width', type=int, default=256,
                        help="input width, default: 256")
    parser.add_argument('--crop-height', type=int, default=224,
                        help="input crop height, default: 224")
    parser.add_argument('--crop-width', type=int, default=224,
                        help="input crop width, default: 224")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    # model
    parser.add_argument('-a', '--arch', type=str, default='',
                        choices=models.names()) 
    parser.add_argument('--model-type', type=str, default='global')
    parser.add_argument('--num-m-features', type=int, default=128)
    parser.add_argument('--masks', type=int, default=20)
    # optimizer
    parser.add_argument('--lr', type=float, default=1e-2,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--lrm', type=float, default=1.0)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--weights', type=str, default='', metavar='PATH')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--step-size', type=int, default=10)
    parser.add_argument('--start-save', type=int, default=0,
                        help="start saving checkpoints after specific epoch")
    parser.add_argument('--warm-up-ep', type=int, default=40)
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
