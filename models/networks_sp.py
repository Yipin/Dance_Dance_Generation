'''
Implementation of semantic segmentation and pose feature loss (SP loss)
Applied pre-trained network proposed by
Mutual Learning to Adapt for Joint Human Parsing and Pose Estimation (https://github.com/NieXC/pytorch-mula)
to compute human segmentation/pose representation.

We thank the authors for providing pre-trained weights.
'''

import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
##
from nets_mula.hourglass_based_network import MuLA_HG_MSRAInit
import torchvision.transforms as transforms

''' define pose and segment loss (SP) '''
class SPLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(SPLoss, self).__init__()

        self.mulanet= Mula_Net().cuda()

        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        x_pose, x_seg = self.mulanet(x)
        y_pose, y_seg = self.mulanet(y)

        # compute losses
        loss_pose = self.criterion(x_pose, y_pose.detach())
        loss_seg = self.criterion(x_seg, y_seg.detach())
        loss = loss_pose + 0.01*loss_seg

        return loss

class Mula_Net(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Mula_Net, self).__init__()

        # build net arch
        self.mula_net = MuLA_HG_MSRAInit()
        pose_net_stride = 4

        # load pre-trained model (only load layers related with feature compute)
        model_path = './pretrained_snapshots/mula_lip.pth.tar'

        checkpoint = torch.load(model_path)
        pretrained_dict = checkpoint['state_dict']
        model_dict = self.mula_net.state_dict()
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        self.mula_net.load_state_dict(pretrained_dict)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        # define mean / std and downsample
        self.mean = 0.5
        self.std = 1
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, X):
        # pre-process data
        # [-1, 1] to [0, 1] to [-0.5, 0.5]
        X = (X + 1.0) / 2.0
        X = (X - self.mean) / self.std
        if X.size()[-1] != 256:
            X = self.downsample(X)

        # run self.mula_net
        pred_pose, pred_seg = self.mula_net(X)
        pred_pose = pred_pose[-1]
        pred_seg = pred_seg[-1]

        return pred_pose, pred_seg
