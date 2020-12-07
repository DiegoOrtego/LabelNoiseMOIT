# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

##Got from: https://github.com/MalongTech/research-xbm
#(CVPR20 paper)

##Doh: Added modification to support mixup

import torch


class XBM:
    def __init__(self, args, device):
        self.K = args.xbm_per_class*args.num_classes*2 # We want to store a minimum number of samples per-class. x2 due to the augmented views
        #self.feats = torch.zeros(self.K, args.low_dim).to(device) # original
        #self.targets = torch.zeros(self.K, dtype=torch.long).to(device) # original
        self.feats = -1.0 * torch.ones(self.K, args.low_dim).to(device) # doh
        self.targets = -1.0 * torch.ones(self.K, dtype=torch.long).to(device) # doh

        self.ptr = 0

    @property
    def is_full(self):
        #return self.targets[-1].item() != 0 #original
        return self.targets[-1].item() != -1 #doh

    def get(self):
        if self.is_full:
            return self.feats, self.targets

        else:
            return self.feats[:self.ptr], self.targets[:self.ptr]


    def enqueue_dequeue(self, feats, targets):
        q_size = len(targets)

        if self.ptr + q_size > self.K:
            self.feats[-q_size:] = feats
            self.targets[-q_size:] = targets
            self.ptr = 0
        else:
            self.feats[self.ptr: self.ptr + q_size] = feats
            self.targets[self.ptr: self.ptr + q_size] = targets
            self.ptr += q_size