import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math

from printt import printt
from .node import Node


def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass


class Leaf(Node):
    mean = 0.5
    std = 0.1

    def __init__(self, index, num_classes):
        super(Leaf, self).__init__(index)
        a = 24  # 15 24
        self.pred = nn.Linear(a, num_classes)  ###1024 // 4
        self.norm_o = nn.LayerNorm(num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, logits, patches, **kwargs):
        print('=================================leaf')
        # input()
        print('patches', patches.shape)  # [4, 768, 24])
        batch_size = patches.size(0)
        node_attr = kwargs.setdefault('attr', dict())
        print('logits', logits.shape)  # ([4, 5])

        node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        x = F.adaptive_max_pool1d(patches.permute(0, 2, 1), 1).squeeze(-1)

        # patches=patches.mean(dim=[1])
        # x = patches

        print('x', x.shape)  # ([4, 24])
        ###

        # print('self.pred', self.pred(x))  # self.pred tensor([[ 0.3943, -0.2017,  0.2177,  0.1529],
        # [ 0.3929, -0.2046,  0.2205,  0.1489]], )
        # dists = self.norm_o(self.pred(x) + logits)  # self.pred(x) + logits)#去除norm效果好
        dists = self.pred(x) + logits
        print('dists', dists.shape)  # ([4, 5])

        # self.dists = F.softmax(dists)###分类
        self.dists = self.softmax(dists)#有softmax效果好
        print('self.dists', self.dists.shape)  # torch.Size([4, 5])

        node_attr[self, 'ds'] = self.dists
        # print('node_attr',node_attr)

        ##
        # input()
        return self.dists, node_attr

    def hard_forward(self, logits, patches, **kwargs):
        return self(logits, patches, **kwargs)

    def explain_internal(self, logits, patches, x_np, node_id, **kwargs):
        return self(logits, patches, **kwargs)

    def explain(self, logits, patches, l_distances, r_distances, x_np, y, prefix, r_node_id, pool_map,
                **kwargs):
        return self(logits, patches, **kwargs)

    @property
    def requires_grad(self) -> bool:
        return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    @property
    def size(self) -> int:
        return 1

    @property
    def leaves(self) -> set:
        return {self}

    @property
    def branches(self) -> set:
        return set()

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self}

    @property
    def num_branches(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 0
