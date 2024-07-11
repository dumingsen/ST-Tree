# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from printt import printt
from .tree_elements import *
from .tree_elements import branch_exp
from .tree_elements.branch_exp import Branch_exp
# from .tree_elements import leaf
from .tree_elements import node
import matplotlib.pyplot as plt
import os

# ===================================================swin transformer=================================================
from model.attention import attentionNet


# ===================================================tree============================================================

class DTree(nn.Module):
    mean = 0.5
    std = 0.1

    ##
    def __init__(self, num_classes, tree_depth, embed_dim, proto_size):
        super(DTree, self).__init__()

        self.proto_size = proto_size
        self.num_classes = num_classes
        self._parents = dict()
        self._root = self._init_tree(num_classes, tree_depth, proto_size)  #
        self._set_parents()
        self.img_size = 224 // 16

        self._out_map = {n.index: i for i, n in zip(range(2 ** (tree_depth) - 1), self.branches)}
        self._maxims = {n.index: float('-inf') for n in self.branches}
        self.num_prototypes = self.num_branches
        self.num_leaves = len(self.leaves)
        self._leaf_map = {n.index: i for i, n in zip(range(self.num_leaves), self.leaves)}
        self.proto_dim = embed_dim // 4

        prototype_shape = [self.num_prototypes, self.proto_dim] + self.proto_size
        self.prototype_shape = prototype_shape
        self.add_on = nn.Sequential(
            nn.Linear(embed_dim, self.proto_dim, bias=False),
            nn.Sigmoid()
        )
        self.attrib_pvec = nn.Parameter(torch.randn(prototype_shape), requires_grad=True)
        self._init_param()

        self.add_on1 = nn.Sequential(
            nn.Linear(embed_dim, self.proto_dim, bias=False),
            nn.Sigmoid()
        )

    ##
    def forward(self, logits, patches, pool_map, attn_weights, **kwargs):
        print('prototype_shape', np.array(self.prototype_shape).shape)  # 3,
        print('self.proto_size', np.array(self.proto_size).shape)  # 1,
        print('Dtree ', patches.shape)
        # logits 分类
        # patches 2 3072 3 特征
        patches = self.add_on(patches)  #

        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        print('self.attrib_pvec.chunk(self.num_prototypes, dim=0)', self.attrib_pvec.shape)  # ([15, 51, 1, 1])
        print('.chunk(self.num_prototypes, dim=0)', self.num_prototypes)  # 15
        # print(kwargs['conv_net_output'].shape)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['attn_weights'] = attn_weights

        ##定义树 self.dists, node_attr
        out, attr = self._root.forward(logits, patches, **kwargs)

        return out

    ##
    def hard_forward(self, logits, patches, pool_map, attn_weights, **kwargs):
        patches = self.add_on(patches)
        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['attn_weights'] = attn_weights

        out, attr = self._root.hard_forward(logits, patches, **kwargs)

        return out

    def explain_internal(self, logits, patches, x_np, pool_map, attn_weights, **kwargs):
        patches = self.add_on(patches)

        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['maxims'] = dict(self._maxims)
        kwargs['img_size'] = self.img_size

        out, attr = self._root.explain_internal(logits, patches, x_np, 0, **kwargs)
        return out

    def explain(self, logits, patches, x_np, y, pool_map, prefix: str, **kwargs):
        print('=================explain', patches.shape)  # [1, 192, 60]
        # patches = patches.reshape(patches.shape[0], -1, patches.shape[2] // 12)
        print('pat', patches.shape)
        # 切换对应通道通道
        patches = self.add_on(patches)
        print('============swin tree patch', patches.shape)  # ([1, 768, 51]) #[1, 768, 206])

        kwargs['conv_net_output'] = self.attrib_pvec.chunk(self.num_prototypes, dim=0)
        kwargs['out_map'] = dict(self._out_map)
        kwargs['img_size'] = self.img_size

        print('prefix', prefix)
        if not os.path.exists(os.path.join('heatmap', prefix)):
            os.makedirs(os.path.join('heatmap', prefix))

        # plt.imsave(fname=os.path.join('heatmap', prefix, 'input_image.png'),
        #            arr=x_np, vmin=0.0, vmax=1.0)

        x = x_np.transpose(1, 0)
        a = range(x_np.shape[0])
        plt.plot(a, x[0], marker='o', markersize=3)
        plt.plot(a, x[1], marker='o', markersize=3)
        plt.plot(a, x[2], marker='o', markersize=3)
        plt.legend(['dim0', 'dim1', 'dim2', ])
        plt.savefig("D:/桌面/NTree-master/heatmap/{0}/filename.png".format(prefix))
        # plt.show()
        plt.close()

        r_node_id = 0
        l_sim = None
        r_sim = None
        out, attr = self._root.explain(logits, patches, l_sim, r_sim, x_np, y, prefix, r_node_id, pool_map,
                                       **kwargs)

        return out

    def get_min_by_ind(self, left_distance, right_distance):
        B, Br, W, H = left_distance.shape

        relative_distance = left_distance / (left_distance + right_distance)
        relative_distance = relative_distance.view(B, Br, -1)
        _, min_dist_idx = relative_distance.min(-1)
        min_left_distance = left_distance.view(B, Br, -1).gather(-1, min_dist_idx.unsqueeze(-1))
        return min_left_distance

    # 树
    def _init_tree(self, num_classes, tree_depth, proto_size):
        def _init_tree_recursive(i, d):
            if d == tree_depth:

                # 叶子
                return Leaf(i, num_classes)
            else:
                left = _init_tree_recursive(i + 1, d + 1)
                right = _init_tree_recursive(i + left.size + 1, d + 1)
                ##分支
                return Branch_exp(i, left, right, proto_size)

        return _init_tree_recursive(0, 0)  ###

    def _init_param(self):
        def init_weights_xavier(m):
            if type(m) == torch.nn.Linear:
                torch.nn.init.xavier_normal_(m.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

        with torch.no_grad():
            torch.nn.init.normal_(self.attrib_pvec, mean=self.mean, std=self.std)
            self.add_on.apply(init_weights_xavier)

    def _set_parents(self):
        def _set_parents_recursively(node):
            if isinstance(node, Branch_exp):
                self._parents[node.r] = node
                self._parents[node.l] = node
                _set_parents_recursively(node.r)
                _set_parents_recursively(node.l)
                return
            if isinstance(node, Leaf):
                return
            raise Exception('Unrecognized node type!')

        self._parents.clear()
        self._parents[self._root] = None
        _set_parents_recursively(self._root)

    @property
    def root(self):
        return self._root

    @property
    def leaves_require_grad(self) -> bool:
        return any([leaf.requires_grad for leaf in self.leaves])

    @leaves_require_grad.setter
    def leaves_require_grad(self, val: bool):
        for leaf in self.leaves:
            leaf.requires_grad = val

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self.backbone.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self.backbone.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    @property
    def depth(self) -> int:
        d = lambda node: 1 if isinstance(node, Leaf) else 1 + max(d(node.l), d(node.r))
        return d(self._root)

    @property
    def size(self) -> int:
        return self._root.size

    @property
    def nodes(self) -> set:
        return self._root.nodes

    @property
    def nodes_by_index(self) -> dict:
        return self._root.nodes_by_index

    @property
    def node_depths(self) -> dict:

        def _assign_depths(node, d):
            if isinstance(node, Leaf):
                return {node: d}
            if isinstance(node, Branch):
                return {node: d, **_assign_depths(node.r, d + 1), **_assign_depths(node.l, d + 1)}

        return _assign_depths(self._root, 0)

    @property
    def branches(self) -> set:
        return self._root.branches

    @property
    def leaves(self) -> set:
        return self._root.leaves

    @property
    def num_branches(self) -> int:
        return self._root.num_branches


# ==================================================总 模 型=================================================
class ViT_NeT(nn.Module):
    def __init__(self, config,
                 t=1,  # 长度
                 down_dim=1024,  # length = 1536 * 2，降维维度

                 hidden_dim=(96, 62),  ##192
                 layers=(2, 2, 6, 2),
                 heads=(3, 6, 12, 24),
                 channels=1,
                 num_classes=1,
                 head_dim=32,
                 window_size=1,
                 downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
                 relative_pos_embedding=True,
                 wa=1,
                 prob=1,
                 mask=1, ):
        super(ViT_NeT, self).__init__()

        self.patch_embedding = attentionNet(
            t=t,  # 长度
            down_dim=down_dim,  # length = 1536 * 2，降维维度
            hidden_dim=hidden_dim,
            layers=layers,

            heads=heads,
            channels=channels,
            num_classes=num_classes,
            head_dim=head_dim,
            window_size=window_size,
            downscaling_factors=downscaling_factors,  # 代表多长的时间作为一个特征
            wa=wa,
            prob=prob,
            mask=mask,

        )
        t = 96  # 96 60
        self.tree = DTree(num_classes=num_classes,  # config.MODEL.NUM_CLASSES,
                          tree_depth=config.MODEL.TREE.DEPTH,
                          embed_dim=t,  # 1024,
                          proto_size=config.MODEL.TREE.PROTO_SIZE)

        self.norm = nn.LayerNorm(t)  ###channels t #注意，实际实验的代码和可解释的代码不同之处
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.head = nn.Linear(t, num_classes) if num_classes > 0 else nn.Identity()  ###channels t
        self.pp = nn.Linear(768, 768 // 6)

    def forward(self, x):  ####
        # trainx
        # torch.Size([2, 1, 3, 206])
        # PatchMerging1
        # torch.Size([2, 3072, 3])
        # PatchMerging2
        # torch.Size([2, 768, 12])
        # PatchMerging3
        # torch.Size([2, 768, 206])
        # attention
        # torch.Size([2, 768, 206])
        # attention
        # torch.Size([2, 768, 206])
        # encoder
        # torch.Size([2, 768, 206])
        x = torch.squeeze(x, dim=1)

        # transformer
        _, x = self.patch_embedding(x)
        print('transformer', x.shape)  # [2, 3072, 3] [2, 768, 206])

        # 1
        patches = self.norm(x)  # B L C  #注意，实际实验的代码和可解释的代码不同之处
        print('patches', patches.shape)  # 2, 768, 206])
        # 2
        x = self.avgpool(patches.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        print('flatten', x.shape)  # [2, 3 ]    [2, 206])
        # 3
        x = self.head(x)
        print('logits x', x.shape)  # [2, 4])

        # tree模型 4
        out = self.tree(x, patches, None, None)  # 分类；特征
        return out

    def hard_forward(self, x):
        logits, patches = self.patch_embedding(x)
        out = self.tree.hard_forward(logits, patches, None, None)
        return out

    def explain(self, x, y, prefix):
        h = x
        print('explain x', x.shape)  # ([1, 206, 3])
        print('y', y.shape)  # 1

        # 可视化
        # ================================================================================
        # ????
        x_np = np.clip(((x + 1) / 2).permute(0, 2, 1)[0].cpu().numpy(), 0., 1.)
        # print('x_np', x_np.shape)  # (3,206)
        # x_np=x.permute(0, 2, 1)

        # if not os.path.exists(os.path.join('origintimechange', str(prefix))):
        #     os.makedirs(os.path.join('origintimechange', str(prefix)))
        #
        # x1 = x_np
        # for a in range(x1.shape[0]):
        #     plt.plot(x1[a])
        #     plt.savefig("D:/桌面/NTree-master/origintimechange/{0}/维度{1}.png".format(prefix, a))
        #     plt.show()
        #     plt.close()
        # input()
        # ================================================================================

        # logits, patches = self.patch_embedding(x)
        _, x = self.patch_embedding(x)
        print('transformer', x.shape)  # [1 768, 206])

        # 1
        patches = self.norm(x)  # B L C
        print('patches', patches.shape)  # 1, 768, 206])
        # 2
        x = self.avgpool(patches.transpose(1, 2))  # B C 1
        print('avgpool', x.shape)  # 1, 206, 1])
        x = torch.flatten(x, 1)
        print('flatten', x.shape)  # )1 206
        # 3
        x = self.head(x)
        print('logits x', x.shape)  # [1, 4])

        # patches = self.pp(patches.permute(0, 2, 1))
        # print(patches.shape)  # ([4, 96, 192])

        h = h.reshape(h.shape[2], h.shape[1])
        printt('h', h.shape)  # torch.Size([3, 206])
       # self.tree.explain(x, patches.permute(0, 2, 1), h, y, None, prefix)  # x_np
        self.tree.explain(x, patches, h, y, None, prefix)

    def explain_internal(self, x):
        x_np = np.clip(((x + 1) / 2).permute(0, 2, 3, 1)[0].cpu().numpy(), 0., 1.)
        logits, patches = self.patch_embedding(x)
        out = self.tree.explain_internal(logits, patches, x_np, None, None)
        return out
# def print(*args, **kwargs):
#     flag = False
#     # flag = False
#     if flag:
#         print(*args, **kwargs)
#     else:
#         pass
