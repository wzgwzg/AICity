from __future__ import print_function, absolute_import
import sys, os
import time
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss
from .utils.meters import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion, logs_dir='.'):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion
        self.logs_dir = logs_dir

    def train(self, epoch, data_loader, optimizer, base_lr, warm_up=False, warm_up_ep=40, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            parts_loss, prec1 = self._forward(inputs, targets, epoch)
            loss = parts_loss[0]
            for part_idx in range(1, len(parts_loss)):
                loss += parts_loss[part_idx]
            loss /= len(parts_loss)

            losses.update(loss.data.item(), targets.size(0))
            precisions.update(prec1, targets.size(0))

            if warm_up: 
                warm_iters = float(len(data_loader) * warm_up_ep)
                if epoch < warm_up_ep:
                    lr = (base_lr / warm_iters) + (epoch*len(data_loader) +(i+1))*(base_lr / warm_iters)
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)
                else:
                    lr = base_lr
                    for g in optimizer.param_groups:
                        g['lr'] = lr * g.get('lr_mult', 1)

            optimizer.zero_grad()
            torch.autograd.backward(parts_loss, [torch.ones(1).cuda() for part_idx in range(len(parts_loss))])
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets, epoch=0):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, num_classes, logs_dir='.'):
        super(Trainer, self).__init__(model, criterion, logs_dir) 
        self.crossentropy_loss = nn.CrossEntropyLoss().cuda()
        self.margins = [0.5]
        self.bh_losses = nn.ModuleList([ TripletLoss(self.margins[i]).cuda() for i in range(len(self.margins)) ])

    def _parse_data(self, inputs):
        assert (len(inputs) == 5 or len(inputs) == 4)
        has_mask = (len(inputs) == 5)
        if has_mask:
            imgs, masks, _, pids, _ = inputs
            inputs = [Variable(imgs), Variable(masks, requires_grad=False)]
        else:
            imgs, _, pids, _ = inputs
            inputs = [Variable(imgs), None]
        targets = Variable(pids.cuda())
        return inputs, targets

    def _forward(self, inputs, targets, epoch=0):
        if isinstance(self.criterion, TripletLoss):
            x_fcs, x_ems = self.model(*inputs)
            fcs_ems_groups = [(x_fcs, x_ems)]
            lw_multi = [1]
            lw_fc_em = [[1, 0.02]]
            losses = []
            for g_i, (fcs, ems) in enumerate(fcs_ems_groups):
                # crossentropy
                fcs_list = list(fcs.split(1, 2)) 
                for i in range(len(fcs_list)):
                    fc = fcs_list[i].contiguous()
                    fc = fc.view(fc.size(0), fc.size(1))
                    loss = self.crossentropy_loss(fc, targets)
                    losses.append(loss * lw_fc_em[g_i][0] * lw_multi[g_i])
                # triplet hard
                ems_list = list(ems.split(1, 2))
                for i in range(len(ems_list)):
                    em = ems_list[i].contiguous()
                    em = em.view(em.size(0), em.size(1))
                    loss_bh = self.bh_losses[g_i](em, targets)
                    losses.append(loss_bh * lw_fc_em[g_i][1] * lw_multi[g_i]) 
            prec = 0
        else:
            raise ValueError("Unsupported loss:", self.criterion)
        return losses, prec
