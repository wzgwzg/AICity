import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

from .senet import *


__all__ = ['aicity_masks_seresnext101']


class AICity_Masks_SeNet(nn.Module):

    def __init__(self, block, layers, num_classes, params):
        super(AICity_Masks_SeNet, self).__init__()
        self.base = SENet(block=block, \
                          layers=layers, \
                          groups=32, \
                          reduction=16, \
                          dropout_p=None, \
                          inplanes=64, \
                          input_3x3=False, \
                          downsample_kernel_size=1, \
                          downsample_padding=0, \
                          last_stride=1)
        self.num_classes = num_classes
        self.num_base_features = 512 * block.expansion
        self.is_initialize = False

        self.num_m_features = params['num_m_features']
        self.num_masks = params['masks']
        self.mask_bn = nn.BatchNorm2d(self.num_masks)

        self.dropout = nn.Dropout()
        self.m_bn = nn.BatchNorm2d(self.num_base_features)
        self.m_dimred_fc = nn.ModuleList([ \
            nn.Linear(self.num_base_features, self.num_m_features, True) \
            for part_idx in range(self.num_masks) \
        ])
        self.m_dimred_bn = nn.ModuleList([ \
            nn.BatchNorm1d(self.num_m_features) \
            for part_idx in range(self.num_masks) \
        ])

        self.m_W = nn.ParameterList([ nn.Parameter(torch.randn(self.num_classes, self.num_m_features), requires_grad=True) \
                    for part_idx in range(self.num_masks) ])
        self.m_b = nn.ParameterList([ nn.Parameter(torch.randn(self.num_classes, 1), requires_grad=True) \
                    for part_idx in range(self.num_masks) ])
        for part_idx in range(self.num_masks):
            nn.init.normal_(self.m_W[part_idx].data, 0, 0.01)
            self.m_b[part_idx].data.zero_()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def save_simmat(self, path):
        W = torch.cat([w for w in self.m_W], 1).contiguous()
        n, c = W.size()
        W = W.div(W.norm(2, 1, keepdim=True).clamp(min=1e-16).expand_as(W))
        W_trans = W.transpose(0, 1).contiguous()
        M = W.mm(W_trans)
        np.save(path, M.data.cpu().numpy())
        return M 

    def set_initialize(self, is_initialize):
        self.is_initialize = is_initialize

    def _net_forward(self, x, masks):
        x = self._backbone_forward(x)
        x = self._head_forward(x, masks)
        return x

    def _backbone_forward(self, x):
        return self.base(x)

    def _head_forward(self, x, masks):
        n = x.size(0); base_c = x.size(1)
        x_mask_pool_parts = self._mask_pool(x, masks)
        x_mask_pool_parts = x_mask_pool_parts.view(n, base_c, -1)

        x_mask_pool_parts = x_mask_pool_parts.split(1, 2)
        x_fcs = []; x_ems = []
        for i in range(self.num_masks):
            x_mask_pool_part = x_mask_pool_parts[i].contiguous().view(n, base_c)
            x_mask_pool_part = self.dropout(x_mask_pool_part)
            x_mask_pool_part = self.m_dimred_fc[i](x_mask_pool_part)
            x_mask_pool_part = self.m_dimred_bn[i](x_mask_pool_part)
            x_mask_em_part = x_mask_pool_part.view(n, self.num_m_features, 1)
            x_ems.append(x_mask_em_part)
            x_mask_fc_part = self._fc(x_mask_pool_part, self.m_W[i], self.m_b[i])
            x_mask_fc_part = x_mask_fc_part.view(n, self.num_classes, 1)
            x_fcs.append(x_mask_fc_part)
        x_fcs = torch.cat(x_fcs, 2)
        x_ems = torch.cat(x_ems, 2)
        
        return x_fcs, x_ems

    def _mask_pool(self, x, masks):
        n, c, h, w = x.size()
        x = self.m_bn(x)
        x = x.view(n, c, h*w)
        masks_trans = masks.view(n, self.num_masks, h*w).transpose(1, 2).contiguous()
        x_mask_pool_parts = x.bmm(masks_trans)
        return x_mask_pool_parts

    def _fc(self, x, W, b):
        x = x.mm(W.transpose(0, 1).contiguous())
        x = x.add(b.transpose(0, 1).contiguous().expand_as(x))
        return x

    def _horizental_flipping(self, x):
        w = x.size(3)
        x_hf = []
        x_w_eles = list(x.split(1, 3))
        for i in range(w):
            x_w_ele = x_w_eles[w-1-i].contiguous()
            x_hf.append(x_w_ele)
        x_hf = torch.cat(x_hf, 3)
        return x_hf

    def _test_forward(self, x, masks):
        _, x_ems = self._net_forward(x, masks)
        x_ems = x_ems.view(x_ems.size(0), -1)
        return x_ems

    def _preprocess_masks(self, masks):
        masks = self.mask_bn(masks) 
        masks = masks.mul(20)
        return masks

    def forward(self, x, masks):
        masks = self._preprocess_masks(masks)

        if self.training:
            x_fcs, x_ems = self._net_forward(x, masks)
            return x_fcs, x_ems
        else:
            if self.is_initialize:
                _, x_ems = self._net_forward(x, masks)
                return x_ems
            else:
                x_org = x
                x_hf = self._horizental_flipping(x)
                masks_org = masks
                masks_hf = self._horizental_flipping(masks)
                x_org = self._test_forward(x_org, masks_org)
                x_hf = self._test_forward(x_hf, masks_hf)
                x = x_org.add(x_hf).div(2)
    
                x = x.view(x.size(0), -1)
                x = x.div(x.norm(2, 1, keepdim=True).add(1e-8).expand_as(x))
                x = x.view(x.size(0), x.size(1), -1)
                 
                return x


def aicity_masks_seresnext101(pretrained=False, **kwargs):
    model = AICity_Masks_SeNet(SEResNeXtBottleneck, [3, 4, 23, 3], **kwargs)
    return model
