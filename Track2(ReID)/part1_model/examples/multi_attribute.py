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
from reid.loss import MultiAttributeLoss, TypeAttributeLoss

from reid import datasets
from reid import models
from reid.dist_metric import DistanceMetric
from reid.attribute_trainers import Multi_Attribute_Trainer, Type_Attribute_Trainer
from reid.evaluators import Evaluator
from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Attribute_Preprocessor
from reid.utils.data.sampler import RandomIdentityAttributeSampler
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint


def get_data(name, split_id, data_dir, big_height, big_width, target_height, target_width, batch_size, num_instances,
        workers, combine_trainval):
    root = osp.join(data_dir, name)

    dataset = datasets.create(name, root, split_id=split_id, download=False)

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_set = dataset.trainval if combine_trainval else dataset.train
    num_classes = (dataset.num_trainval_ids if combine_trainval
                   else dataset.num_train_ids)

    train_transformer = T.Compose([
        T.ResizeRandomCrop(big_height, big_width, target_height, target_width),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalizer,
        T.RandomErasing(0.5),
    ])

    test_transformer = T.Compose([
        T.RectScale(target_height, target_width),
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


def main(args):
    print(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.benchmark = True

    # Redirect print to both console and log file
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))

    # Create data loaders
    if args.big_height is None or args.big_width is None or args.target_height is None or args.target_width is None:
        args.big_height, args.big_width, args.target_height, args.target_width = (256, 256, 224, 224)
    dataset, num_classes, train_loader, test_loader = \
        get_data(args.dataset, args.split, args.data_dir, args.big_height, args.big_width,
                 args.target_height, args.target_width, args.batch_size, args.num_instances,
                 args.workers, args.combine_trainval)

    # Create model
    model = models.create(args.arch, num_classes=num_classes, num_features=args.features, is_cls=args.is_cls)

    # Load from checkpoint
    start_epoch = best = 0
    if args.weights: 
        #model_dict = model.state_dict()
        #checkpoint_load = {k: v for k, v in (checkpoint['state_dict']).items() if k in model_dict}
        #model_dict.update(checkpoint_load)
        #model.load_state_dict(model_dict)
        if args.arch == 'cross_trihard_senet101':        
            model.base.load_param(args.weights)
        else:
            checkpoint = load_checkpoint(args.weights)
            del(checkpoint['fc.weight'])
            del(checkpoint['fc.bias'])
            model.base.load_state_dict(checkpoint)
    if args.resume and not args.weights:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        print("=> Start epoch {}  best top1 {:.1%}"
              .format(start_epoch, best))
    
    model = nn.DataParallel(model).cuda()

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test:")
        evaluator.evaluate(test_loader, dataset.query, dataset.gallery)
        return

    # Criterion
    ranking_loss = nn.MarginRankingLoss(margin=args.margin).cuda()
    if args.multi_attribute == 1:
        criterion = { 'MultiAttributeLoss': MultiAttributeLoss(is_cls=args.is_cls).cuda(), \
                  'trihard': TripletLoss(ranking_loss).cuda() }
    else:
        criterion = { 'TypeAttributeLoss': TypeAttributeLoss(is_cls=args.is_cls).cuda(), \
                  'trihard': TripletLoss(ranking_loss).cuda() }


    # Optimizer
    if hasattr(model.module, 'base'):
        base_params = []
        base_bn_params = []
        for name, p in model.module.base.named_parameters():
            if 'bn' in name:
                base_bn_params.append(p)
            else:
                base_params.append(p)
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': base_params, 'lr_mult': 0.1},
            {'params': base_bn_params, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': args.lr_mult}]
    else:
        param_groups = model.parameters()

    if args.optimizer == 0:
	print('SGD')
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else:
        print('Adam')
        optimizer = torch.optim.Adam(params=param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # Trainer
    if args.multi_attribute == 1:
        print('Multi Attribute')
        trainer = Multi_Attribute_Trainer(model, criterion, metric_loss_weight=args.metric_loss_weight, sub_task_loss_weight=args.sub_task_loss_weight)
    else:
        print('Type Attribute')
        trainer = Type_Attribute_Trainer(model, criterion, metric_loss_weight=args.metric_loss_weight, sub_task_loss_weight=args.sub_task_loss_weight)

    # Schedule learning rate
    def adjust_lr(epoch):
	step_size, step_size2, step_size3 = args.step_size, args.step_size2, args.step_size3 
        if epoch <= step_size:
            lr = args.lr
        elif epoch <= step_size2:
            lr = args.lr * 0.1 
        elif epoch <= step_size3:
            lr = args.lr * 0.01
        else:
            lr = args.lr * 0.001
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)
        return lr

    # Start training
    for epoch in range(start_epoch+1, args.epochs+1):
        lr = adjust_lr(epoch)
        trainer.train(epoch, train_loader, optimizer, lr, warm_up=True, warm_up_ep=args.warm_up_ep)
        if epoch % args.epoch_inter == 0 or epoch >=args.dense_evaluate:
            tmp_res = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, is_attribute=True)
            if tmp_res > best and epoch >= args.start_save:
                best = tmp_res
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch,
                }, False, fpath=osp.join(args.logs_dir, 'pass%d.pth.tar' % (epoch)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cross_entropy_trihard_net")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--big_height', type=int, default=256,
                        help="input height, default: 256")
    parser.add_argument('--big_width', type=int, default=256,
                        help="input width, default: 256")
    parser.add_argument('--target_height', type=int, default=224,
                        help="crop height, default: 224")
    parser.add_argument('--target_width', type=int, default=224,
                        help="crop width, default: 224")
    parser.add_argument('--combine-trainval', action='store_true',
                        help="train and val sets together for training, "
                             "val set alone for validation")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 4")
    parser.add_argument('--step_size', type=int, default=400)
    parser.add_argument('--step_size2', type=int, default=700)
    parser.add_argument('--step_size3', type=int, default=750)
    # model
    parser.add_argument('-a', '--arch', type=str, default='cross_entropy_trihard_resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--multi_attribute', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    parser.add_argument('--is_cls', type=int, default=1)
    # optimizer
    parser.add_argument('--optimizer', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--lr_mult', type=float, default=1.0)
    parser.add_argument('--metric_loss_weight', type=float, default=0.02)
    parser.add_argument('--sub_task_loss_weight', type=float, default=0.5)
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
    parser.add_argument('--epoch_inter', type=int, default=10)
    parser.add_argument('--dense_evaluate', type=int, default=90)
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
