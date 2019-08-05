'''
Derived from aligned_dataset.py

This code is used to preprocess data for training / testing.

foreground stage:
- trans_segs [H, W, 30] uint8 (transformed body part segments)
- ref_pose [H, W, 10], float32 [0, 1] (reference pose map representation)
- ref_frames_foreground_g [H, W, 3] uint8 (foreground stage supervision)
fusion stage:
- background_image [H, W, 3] uint8 (fixed background image)
- ref_frames [H, W, 3] uint8 (fusion stage supervision)
'''
from __future__ import unicode_literals

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
##
import sys
import numpy as np
import torch



def preprocess(image):
    # [0, 1] => [-1, 1]
    return image * 2.0 - 1.0


def deprocess(image):
    # [-1, 1] => [0, 1]
    return (image + 1.0) / 2.0

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        ### input A (trans_segs)
        dir_A = '_A' if self.opt.label_nc == 0 else 'trans_segs'
        self.dir_A = os.path.join(opt.dataroot, dir_A)
        self.A_paths = sorted(make_dataset(self.dir_A))

        ### input B (ref_poses)
        ### *_pointer records path to the data tensor
        dir_B = '_B' if self.opt.label_nc == 0 else 'ref_poses_polymark_pointer'
        self.dir_B = os.path.join(opt.dataroot, dir_B)
        self.B_paths = sorted(make_dataset(self.dir_B))

        ### input C (background_image)
        dir_C = '_C' if self.opt.label_nc == 0 else 'background_images'
        self.dir_C = os.path.join(opt.dataroot, dir_C)
        self.C_paths = sorted(make_dataset(self.dir_C))

        ### input D (ref_frames_foreground_g)
        dir_D = '_D' if self.opt.label_nc == 0 else 'ref_frames_foreground_g_pointer'
        self.dir_D = os.path.join(opt.dataroot, dir_D)
        self.D_paths = sorted(make_dataset(self.dir_D))

        ### input E (ref_frames)
        dir_E = '_E' if self.opt.label_nc == 0 else 'ref_frames_pointer'
        self.dir_E = os.path.join(opt.dataroot, dir_E)
        self.E_paths = sorted(make_dataset(self.dir_E))

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):
        ### input A (trans_segs)
        A_path = self.A_paths[index]
        A = np.load(A_path)
        if self.opt.label_nc == 0:
            sys.exit("label_nc == 0")
        else:
            # [0, 255] -> [0, 1] -> [-1, 1]
            A = A.astype(np.float32) / 255.0
            A = preprocess(A)
            # [H, W, 30] to [30, H, W]
            A = np.transpose(A, (2, 0, 1))

        ### input B (ref_poses_pointer)
        # B is pointer (the path) , load prepre/pre/cur (K = 3) all together
        # TODO: make K a variable
        B_path = self.B_paths[index]
        try:
            B_pointer = '{}'.format(str(np.load(B_path), 'utf-8'))  # python 3.x
        except TypeError:
            B_pointer = '{}'.format(np.load(B_path))  # python 2.x
        B_root = B_pointer[:-19]
        B_prepre_path = '%s/frame_%08d.npy' % (B_root, int(B_pointer[-11:-4])-2)
        B_pre_path = '%s/frame_%08d.npy' % (B_root, int(B_pointer[-11:-4])-1)

        # load B tensor
        # [0, 1] -> [-1, 1]
        B = np.load(B_pointer)
        B = B.astype(np.float32)
        B = preprocess(B)
        # [H, W, 15] -> [15, H, W]
        B = np.transpose(B, (2, 0, 1))

        # load B pre tensor
        if not os.path.isfile(B_pre_path):
            B_pre_path = B_pointer
        B_pre = np.load(B_pre_path)
        B_pre = B_pre.astype(np.float32)
        B_pre = preprocess(B_pre)
        B_pre = np.transpose(B_pre, (2, 0, 1))

        # load B prepre tensor
        if not os.path.isfile(B_prepre_path):
            B_prepre_path = B_pre_path
        B_prepre = np.load(B_prepre_path)
        B_prepre = B_prepre.astype(np.float32)
        B_prepre = preprocess(B_prepre)
        B_prepre = np.transpose(B_prepre, (2, 0, 1))

        # concate [15*3, H, W]
        B = np.concatenate((B_prepre, B_pre, B), axis=0)

        ### input C (background_image)
        C_path = self.C_paths[index]
        C = Image.open(C_path)
        # [0, 255] -> [0, 1]
        C = np.array(C).astype(np.float32) / 255.0
        # [0, 1] -> [-1, 1]
        C = preprocess(C)
        # [H, W, 3] -> [3, H, W]
        C = np.transpose(C, (2, 0, 1))

        D = E = 0
        ### input D (ref_frames_foreground_g)
        D_path = self.D_paths[index]
        try:
            D_pointer = '{}'.format(str(np.load(D_path),'utf-8'))
        except TypeError:
            D_pointer = '{}'.format(np.load(D_path))
        D = np.load(D_pointer)
        # [0, 255] -> [0, 1] -> [-1, 1]
        D = D.astype(np.float32) / 255.0
        D = preprocess(D)
        # [H, W, 3] -> [3, H, W]
        D = np.transpose(D, (2, 0, 1))

        ### input E (ref_frames)
        E_path = self.E_paths[index]
        try:
            E_pointer = '{}'.format(str(np.load(E_path),'utf-8'))
        except TypeError:
            E_pointer = '{}'.format(np.load(E_path))
        E = np.load(E_pointer)
        # [0, 255] -> [0, 1] -> [-1, 1]
        E = E.astype(np.float32) / 255.0
        E = preprocess(E)
        # [H, W, 3] -> [3, H, W]
        E = np.transpose(E, (2, 0, 1))


        input_dict = {'trans_segs': A, 'ref_poses': B, 'background_image': C,
                      'ref_frames_foreground_g': D, 'ref_frames': E, 'path': A_path}

        return input_dict

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'