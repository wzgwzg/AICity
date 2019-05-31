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

from reid import models
from reid.dist_metric import DistanceMetric

from reid.direct_trainers import Direct_Trainer
from reid.direct_evaluators import Evaluator

from reid.utils.data import transforms as T
from reid.utils.data.preprocessor import Direct_Preprocessor
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint
import re
import pdb


def get_data(data_dir, big_height, big_width, target_height, target_width, batch_size, workers, is_train=True):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    train_transformer = T.Compose([
        T.ResizeRandomCrop(big_height, big_width, target_height, target_width),
        T.ToTensor(),
        normalizer,
    ])

    test_transformer = T.Compose([
        T.RectScale(target_height, target_width),
        T.ToTensor(),
        normalizer,
    ])  
    
    if is_train:
	transformer = train_transformer
    else:
	transformer = test_transformer
 	
    data_loader = DataLoader(
        Direct_Preprocessor(data_dir=data_dir,
                     transform=transformer, is_train=is_train),
        batch_size=batch_size, num_workers=workers,
        shuffle=is_train, pin_memory=True)
    
    return data_loader


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
    
    train_loader = \
        get_data(args.train_dir, args.big_height, args.big_width,
                 args.target_height, args.target_width, args.batch_size, args.workers)
    
    test_loader = get_data(args.test_dir, args.big_height, args.big_width,
                 args.target_height, args.target_width, args.batch_size, args.workers, is_train=False)

    # Create model
    model = models.create(args.arch, num_classes=args.num_classes, num_features=args.features)

    # Load from checkpoint
    start_epoch = best = 0
    if args.weights: 
        checkpoint = load_checkpoint(args.weights)
        if args.arch == 'cross_trihard_senet101':        
            del(checkpoint['last_linear.weight'])
            del(checkpoint['last_linear.bias'])
            model.base.load_state_dict(checkpoint)
            #model.base.load_param(args.weights)
        elif args.arch == 'cross_trihard_mobilenet':
            del(checkpoint['state_dict']['module.fc.weight'])
            del(checkpoint['state_dict']['module.fc.bias'])
            model_dict = model.state_dict()
            checkpoint_load = {k.replace('module', 'base'): v for k, v in (checkpoint['state_dict']).items()}
            model_dict.update(checkpoint_load)
            model.load_state_dict(model_dict)
        elif args.arch == 'cross_trihard_shufflenetv2':
            del(checkpoint['classifier.0.weight'])
            del(checkpoint['classifier.0.bias'])
            model.base.load_state_dict(checkpoint)
        elif args.arch == 'cross_trihard_densenet121':
            del(checkpoint['classifier.weight'])
            del(checkpoint['classifier.bias'])
            pattern = re.compile(
        	r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(checkpoint.keys()):
                res = pattern.match(key)
                if res:
            	    new_key = res.group(1) + res.group(2)
            	    checkpoint[new_key] = checkpoint[key]
            	    del checkpoint[key]
	    model.base.load_state_dict(checkpoint)
        elif args.arch == 'cross_trihard_vgg19bn':
            del(checkpoint['classifier.6.weight'])
            del(checkpoint['classifier.6.bias'])
            model.base.load_state_dict(checkpoint)
        #elif args.arch == 'resfpnnet101':
        #    pass
        else:
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

    # Criterion
    criterion = nn.CrossEntropyLoss().cuda()

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
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    else:
        print('Adam')
        optimizer = torch.optim.Adam(params=param_groups, lr=args.lr, weight_decay=args.weight_decay)

    # Trainer
    trainer = Direct_Trainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        step_size, step_size2, step_size3 = args.step_size, args.step_size2, args.step_size3 
        #lr = args.lr * (0.1 ** (epoch // step_size))
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
            tmp_res = evaluator.evaluate(test_loader)
            print('tmp_res: ', tmp_res)
            print('best: ', best)
            if tmp_res > best and epoch >= args.start_save:
                best = tmp_res
                save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch,
                }, False, fpath=osp.join(args.logs_dir, 'pass%d.pth.tar' % (epoch)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="cross_entropy_trihard_net")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('--train_dir', type=str, default='./examples/data/aicity_train_direction')
    parser.add_argument('--test_dir', type=str, default='./examples/data/aicity_train_direction')
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
    parser.add_argument('--step_size', type=int, default=90)
    parser.add_argument('--step_size2', type=int, default=90)
    parser.add_argument('--step_size3', type=int, default=90)
    # model
    parser.add_argument('-a', '--arch', type=str, default='cross_entropy_trihard_resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=1024)
    parser.add_argument('--num_classes', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--margin', type=float, default=0.5,
                        help="margin of the triplet loss, default: 0.5")
    # optimizer
    parser.add_argument('--optimizer', type=int, default=0)
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
