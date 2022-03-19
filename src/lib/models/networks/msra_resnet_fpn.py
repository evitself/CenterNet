# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from typing import List

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1, l1_in, l1_out = self._make_layer(block, 64, layers[0])
        self.layer2, l2_in, l2_out = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3, l3_in, l3_out = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4, l4_in, l4_out = self._make_layer(block, 512, layers[3], stride=2)

        # used for deconv layers
        # self.deconv_layers = self._make_deconv_layer(
        #     3,
        #     [256, 256, 256],
        #     [4, 4, 4],
        # )
        self.deconv_layer1 = self._make_deconv_layer_one(l4_out, 256, 4)
        self.deconv_layer2 = self._make_deconv_layer_one(256, 256, 4)
        self.deconv_layer3 = self._make_deconv_layer_one(256, 256, 4)
        self.deconv_layers = [
            self.deconv_layer1, self.deconv_layer2, self.deconv_layer3
        ]
        # self.final_layer = []

        self.layer3_projection = self._make_fpn_projection_layer(l3_out, 256)
        self.layer2_projection = self._make_fpn_projection_layer(l2_out, 256)
        self.layer1_projection = self._make_fpn_projection_layer(l1_out, 256)
        self.projection_layers = [
            self.layer3_projection, self.layer2_projection, self.layer1_projection
        ]

        for head in sorted(self.heads):
            num_output = self.heads[head]
            if head_conv > 0:
                fc = nn.Sequential(
                    nn.Conv2d(256, head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, num_output,
                              kernel_size=1, stride=1, padding=0))
            else:
                fc = nn.Conv2d(
                    in_channels=256,
                    out_channels=num_output,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            self.__setattr__(head, fc)

        # self.final_layer = nn.ModuleList(self.final_layer)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        layer_in_ch = self.inplanes
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        layer_out_ch = self.inplanes
        return nn.Sequential(*layers), int(layer_in_ch), int(layer_out_ch)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels) -> List[torch.nn.Sequential]:
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        deconv_blocks = []
        for i in range(num_layers):
            layers = []
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes
            deconv_blocks.append(nn.Sequential(*layers))

        return deconv_blocks

    def _make_deconv_layer_one(self, in_channels, out_channels, kernel):
        kernel, padding, output_padding = self._get_deconv_cfg(kernel, 0)
        layers = []
        layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=2,
                padding=padding,
                output_padding=output_padding,
                bias=self.deconv_with_bias))
        layers.append(nn.BatchNorm2d(out_channels, momentum=BN_MOMENTUM))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)

    def _make_fpn_projection_layer(self, in_plains, out_plains):
        layers = [
            nn.Conv2d(
                in_channels=in_plains,
                out_channels=out_plains,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=self.deconv_with_bias
            )]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        l1 = self.layer1(x)
        p1 = self.layer1_projection(l1)

        l2 = self.layer2(l1)
        p2 = self.layer2_projection(l2)

        l3 = self.layer3(l2)
        p3 = self.layer3_projection(l3)

        l4 = self.layer4(l3)

        d1 = self.deconv_layer1(l4)
        d2 = self.deconv_layer2(d1 + p3)
        d3 = self.deconv_layer3(d2 + p2)

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(d3 + p1)
        return [ret]

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for deconv_layer in self.deconv_layers:
                for _, m in deconv_layer.named_modules():
                    if isinstance(m, nn.ConvTranspose2d):
                        # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        nn.init.normal_(m.weight, std=0.001)
                        if self.deconv_with_bias:
                            nn.init.constant_(m.bias, 0)
                    elif isinstance(m, nn.BatchNorm2d):
                        # print('=> init {}.weight as 1'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        nn.init.constant_(m.weight, 1)
                        nn.init.constant_(m.bias, 0)
            for proj_layer in self.projection_layers:
                for _, m in proj_layer.named_modules():
                    if isinstance(m, nn.Conv2d):
                        nn.init.normal_(m.weight, std=0.001)
                        if self.deconv_with_bias:
                            nn.init.constant_(m.bias, 0)
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
                final_layer = self.__getattr__(head)
                for i, m in enumerate(final_layer.modules()):
                    if isinstance(m, nn.Conv2d):
                        # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                        # print('=> init {}.bias as 0'.format(name))
                        if m.weight.shape[0] == self.heads[head]:
                            if 'hm' in head:
                                nn.init.constant_(m.bias, -2.19)
                            else:
                                nn.init.normal_(m.weight, std=0.001)
                                nn.init.constant_(m.bias, 0)
            # pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv):
    block_class, layers = resnet_spec[num_layers]

    model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers, pretrained=True)
    return model
