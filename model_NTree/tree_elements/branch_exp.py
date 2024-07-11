import argparse
import copy
import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from .node import Node
import cv2
import matplotlib.pyplot as plt
import os
from math import sqrt
from timm.models.layers import trunc_normal_
from printt import printt


def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass


def find_high_activation_crop(mask, threshold):
    # 3,206
    print('mask', mask.shape)
    threshold = 1. - threshold  # 1. - threshold
    lower_y, upper_y, lower_x, upper_x = 0, 0, 0, 0
    # for i in range(mask.shape[0]):
    #     if np.amax(mask[i]) > threshold:
    #         lower_y = i
    #         break
    # for i in reversed(range(mask.shape[0])):
    #     if np.amax(mask[i]) > threshold:
    #         upper_y = i
    #         break
    for j in range(mask.shape[1]):
        if np.amax(mask[:, j]) > threshold or np.amax(mask[:, j]) != 0:
            lower_x = j
            break
    for j in reversed(range(mask.shape[1])):
        if np.amax(mask[:, j]) > threshold or np.amax(mask[:, j]) != 0:
            upper_x = j
            break
    return lower_y, upper_y + 1, lower_x, upper_x + 1  # lower_y, upper_y + 1, lower_x, upper_x + 1


class GCB(nn.Module):
    mean = 0.5
    std = 0.1

    def __init__(self, dim, patch_dim=49):
        super(GCB, self).__init__()

        exp_dim = int(dim * 1.)

        self.cm = nn.Linear(dim, 1)
        self.wv1 = nn.Linear(dim, exp_dim)
        self.norm = nn.LayerNorm(exp_dim)
        self.gelu = nn.GELU()
        self.wv2 = nn.Linear(exp_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, patches):
        h = patches
        x = self.cm(patches)
        x = torch.bmm(h.permute(0, 2, 1), F.softmax(x, 1)).squeeze(-1)
        x = self.wv1(x)
        x = self.gelu(self.norm(x))
        x = self.wv2(x)
        x = h + x.unsqueeze(1)
        x = self.ffn_norm(x)
        # x = F.sigmoid(x)
        x = self.sigmoid(x)
        return x


# CBMA  通道注意力机制和空间注意力机制的结合
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # 平均池化高宽为1
        self.max_pool = nn.AdaptiveMaxPool1d(1)  # 最大池化高宽为1

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        kernel_size = 7
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        print('maxout', max_out.shape)  # [4, 768, 1]
        # out = avg_out + max_out  # 加和操作

        out = torch.cat([avg_out, max_out], dim=2)
        out = self.conv1(out.permute(0, 2, 1))

        print('通道注意力', out.shape)  # torch.Size([4, 768, 1])
        return self.sigmoid(out.permute(0, 2, 1))  # sigmoid激活操作


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w,返回索引
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        print('x', x.shape)  # [4, 2, 24]

        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        print('空间注意力', x.shape)  # torch.Size([4, 1, 24])
        return self.sigmoid(x)  # sigmoid激活操作


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        print('cbam', x.shape)
        x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
        print('cbam', x.shape)

        # x1 = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        # print('cbam', x.shape)
        # x2 = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
        # print('cbam', x.shape)
        # x3 = x1 + x2
        # input()
        return x


class Branch_exp(Node):
    mean = 0.5
    std = 0.1

    def __init__(self, index, l: Node, r: Node, proto_size: list):
        super(Branch_exp, self).__init__(index)
        self.l = l
        self.r = r
        self.img_size = 448
        self.gcb_l = GCB(24)  # 258
        self.gcb_r = GCB(24)  # 24 15
        self.max_score = float('-inf')
        self.proto_size = proto_size
        a = 768
        self.cbaml = cbamblock(a, 16, 7)
        self.cbamr = cbamblock(a, 16, 7)

        # 192
        a = 5
        b = 192
        self.linear = nn.Linear(b, 206)
        self.conv = nn.Conv1d(a, 3, 1, bias=False)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, logits, patches, **kwargs):
        print('===========================branch')
        print('patches', patches.shape)  # 2, 768, 51]
        # 拉平， 特征
        batch_size = patches.size(0)

        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g(**kwargs)  # 1, 51, 1, 1])
        print('ps1', ps.shape)  # [1, 51, 1

        ###
        distance = self._l2_conv(patches, ps, stride=1, dilation=1, padding=0)
        print('distane', distance.shape)  # 2 1 768
        similarity = torch.log((distance + 1) / (distance + 1e-4))
        print('sim', similarity.shape)  # 2 1 768

        # self.maxim = F.adaptive_max_pool2d(similarity, (1, 1)).squeeze(-1).squeeze(-1)
        self.maxim = F.adaptive_max_pool1d(similarity, 1).squeeze(-1)
        print('self.maxim', self.maxim.shape)  # 2，1
        to_left = self.maxim[:, 0]
        to_right = 1 - to_left
        print('left right', to_left, to_right)
        # tensor([0.5134, 0.5168], grad_fn=<SelectBackward0>) tensor([0.4866, 0.4832], grad_fn=<RsubBackward1>)

        l_dists, _ = self.l.forward(logits, self.cbaml(patches), **kwargs)  # gcb_l
        print('l_dists', l_dists.shape)  # [2, 4])
        r_dists, _ = self.r.forward(logits, self.cbamr(patches), **kwargs)
        print('r_dists', r_dists.shape)  # [2, 4])
        # python run_UEA.py

        if torch.isnan(self.maxim).any() or torch.isinf(self.maxim).any():
            raise Exception('Error: NaN/INF values!', self.maxim)

        node_attr[self, 'ps'] = self.maxim
        node_attr[self.l, 'pa'] = to_left * pa
        node_attr[self.r, 'pa'] = to_right * pa
        return to_left.unsqueeze(-1) * l_dists + to_right.unsqueeze(-1) * r_dists, node_attr

    def extract_internal_patch(self, similarities, x_np, node_id):
        distance_batch = similarities[0][0]

        distance_batch[distance_batch < distance_batch.max()] = distance_batch.min()
        sz, _ = distance_batch.shape
        distance_batch = distance_batch.view(sz, sz, 1)
        similarity_map = distance_batch.detach().cpu().numpy()

        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map = rescaled_sim_map / np.amax(rescaled_sim_map)
        similarity_heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_sim_map), cv2.COLORMAP_JET)
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[..., ::-1]
        masked_similarity_map = np.ones(similarity_map.shape)
        masked_similarity_map[similarity_map < np.max(
            similarity_map)] = 0  # mask similarity map such that only the nearest patch z* is visualized

        upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                 dsize=(self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)

        high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, 0.98)
        high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
                         high_act_patch_indices[2]:high_act_patch_indices[3], :]

        plt.imsave(
            fname=f'node_interp/node-{node_id}-patch.png',
            arr=high_act_patch, vmin=0.0, vmax=1.0)

        imsave_with_bbox(
            fname=f'node_interp/node-{node_id}.png',
            img_rgb=x_np,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3], color=(0, 0, 255))

    def explain_internal(self, logits, patches, x_np, node_id, **kwargs):
        batch_size = patches.size(0)

        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g(**kwargs)

        distance = self._l2_conv(patches, ps, stride=1, dilation=1, padding=0)
        similarity = torch.log((distance + 1) / (distance + 1e-4))

        self.maxim = F.adaptive_max_pool2d(similarity, (1, 1)).squeeze(-1).squeeze(-1)
        to_left = self.maxim[:, 0]
        to_right = 1 - to_left

        out_map = kwargs['out_map']

        score = self.maxim.squeeze().item()
        if score >= self.max_score:
            self.max_score = score
            similarity = F.conv2d(similarity, weight=torch.ones((1, 1, 2, 2)).cuda(),
                                  stride=2) if self.proto_size == [1, 1] else similarity
            self.extract_internal_patch(similarity, x_np, node_id)

        l_dists, _ = self.l.explain_internal(logits, self.gcb_l(patches), x_np, node_id * 2 + 1, **kwargs)
        r_dists, _ = self.r.explain_internal(logits, self.gcb_r(patches), x_np, node_id * 2 + 2, **kwargs)

        if torch.isnan(self.maxim).any() or torch.isinf(self.maxim).any():
            raise Exception('Error: NaN/INF values!', self.maxim)

        node_attr[self, 'ps'] = self.maxim
        node_attr[self.l, 'pa'] = to_left * pa
        node_attr[self.r, 'pa'] = to_right * pa
        return to_left.unsqueeze(-1) * l_dists + to_right.unsqueeze(-1) * r_dists, node_attr

    def _l2_conv(self, x, proto_vector, tm,stride, dilation, padding,):
        B, L, C = x.shape
        printt('branch', x.shape)  # [2, 768, 51] #[1, 5, 192]) ([1, 128, 24])

        W = int(sqrt(L))

        ###
        printt('tm',tm.shape)
        x = x.permute(0, 2, 1)  # .view(B, C, W)#(B, C, W, W)

        x = x.reshape(x.shape[0], x.shape[1], tm.shape[0], -1)#3 通道修改

        _, _, k = proto_vector.shape  ##
        print('k', k)
        input = torch.rand(3, proto_vector.shape[1], 1)
        print('branch proto_vector.shape', proto_vector.shape)  # 1 24 1
        ones = torch.ones_like(proto_vector, device=x.device).unsqueeze(dim=-1)  # proto_vector
        printt('ones', ones.shape)  # ([1, 51, 1]) [3, 51, 1]) 1, 24, 1,

        x2 = x ** 2
        printt('branch x2', x2.shape)  # [2, 51, 768]) [1, 24, 128])
        # conv1d
        # x2_patch_sum = F.conv2d(x2, weight=ones, stride=stride, dilation=dilation, padding=padding)
        x2_patch_sum = F.conv2d(x2, weight=ones, stride=stride, dilation=dilation, padding=padding)  # 出现相同参数 stride
        # x2_patch_sum = self.conv(x2)
        print('x2_patch_sum', x2_patch_sum.shape)  # 1, 1, 768]) [1, 3, 768]

        p2 = proto_vector ** 2
        p2 = torch.sum(p2, dim=(1, 2))  # 1 2 3
        p2_reshape = p2.view(-1, 1,1)  # , 1

        proto_vector = proto_vector.unsqueeze(dim=-1)
        # xp = F.conv2d(x, weight=proto_vector, stride=stride, dilation=dilation, padding=padding)
        xp = F.conv2d(x, weight=proto_vector, stride=stride, dilation=dilation, padding=padding)#
        print('xp', xp.shape)  # 1, 1, 768])
        intermediate_result = -1 * xp + p2_reshape  # -2
        printt('intermediate_result', intermediate_result.shape)  # 1, 1, 768])([1, 1, 3, 256])

        distances = torch.sqrt(F.relu(x2_patch_sum + intermediate_result))
        printt('dis', distances.shape)  # 1, 1, 768] [1, 3, 768])
        return distances

    def g(self, **kwargs):
        out_map = kwargs['out_map']  # Obtain the mapping from decision nodes to conv net outputs
        conv_net_output = kwargs['conv_net_output']  # Obtain the conv net outputs
        out = conv_net_output[out_map[self.index]]  # Obtain the output corresponding to this decision node
        print('g out', out.shape)  # 1, 51, 1, 1]) [1, 5, 1])
        return out.squeeze(dim=1)

    def plot_map(self, node_id, similarities, prefix, r_node_id, x_np, pool_map, direction, correctness):
        printt('similarities', similarities.shape)#[1, 1, 2, 384]
        # similarities = self.linear(similarities)  # [1, 3, 206] ([1, 1, 3, 64])
        # print('similarities', similarities.shape)

        # max = nn.max(similarities)
        # print('max', max)
        index = np.unravel_index(similarities.detach().numpy().argmax(), similarities.detach().numpy().shape)
        printt('index',index)#(0, 0, 0, 184)

        for i in range(1):
            print('==========================plot map', similarities.shape)
            distance_batch = similarities[0][0][index[2]]  # [1, 1, 768])[1, 3, 768])
            # print(distance_batch - 0.37)  # 768,# 206

            # distance_batch = F.relu(distance_batch)
            distance_batch[distance_batch < 0.37 * distance_batch.max()] = distance_batch.min()
            print(distance_batch.shape)
            sz = distance_batch.shape  # 两个参数 ,_
            print('sz', sz)  # [768] [206]
            distance_batch = distance_batch.view(sz, sz, 1)
            printt('distance_batch', distance_batch.shape)  # [768] 256

            distance_batch = F.relu(distance_batch)

            similarity_map = distance_batch.detach().cpu().numpy()

            printt(similarity_map.shape)  # 256,

            # =====归一化
            rescaled_sim_map = similarity_map - np.amin(similarity_map)
            rescaled_sim_map = rescaled_sim_map / np.amax(rescaled_sim_map)
            printt('=====================rescaled_sim_map', rescaled_sim_map.shape)  # 256,

            # similarity_map = self.upsample2(distance_batch.unsqueeze(dim=0).unsqueeze(dim=0))
            # printt('upsample sim', similarity_map.shape)#[1, 1, 512])

            # similarity_map = similarity_map.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            printt('upsample sim', similarity_map.shape)  # (512,)

            printt(similarity_map.shape)  # (512,)
            printt(np.max(similarity_map))  # 0.3356779
            printt(np.min(similarity_map))  # 0.2767667

            time = x_np  # .reshape(x_np.shape[1], x_np.shape[0])
            printt('time', time.shape)  # (3, 206)

            # for a in range(time.shape[0]):
            #     plt.plot(time[a])
            #     plt.savefig("D:/桌面/NTree-master/heatmap/{0}/维度{1}.png".format(prefix,  a))
            #     plt.show()
            #     plt.close()
            # input()

            # for a in range(time.shape[0]):
            #     plt.plot(time[a])
            #     plt.savefig("D:/桌面/NTree-master/heatmap/{0}/维度{1}.png".format(prefix,  a))
            # plt.show()
            # plt.close()
            # input()

            upsampled_prototype_pattern = cv2.resize(similarity_map,  # masked_similarity_map
                                                     time[0:1, :].shape,
                                                     # dsize=(self.img_size, self.img_size),
                                                     )  # INTER_CUBIC INTER_LANCZOS4 INTER_AREA INTER_LINEAR INTER_NEAREST
            printt('upsampled_prototype_pattern', upsampled_prototype_pattern.shape)  # (206, 1)

            # =============
            upsampled_prototype_pattern = torch.from_numpy(upsampled_prototype_pattern)
            index1 = np.unravel_index(upsampled_prototype_pattern.detach().numpy().argmax(),
                                      upsampled_prototype_pattern.detach().numpy().shape)
            printt(index1)  # (119, 0)

            a = time.shape[1]
            length = a
            b = math.floor(a / 32)  # 16
            printt('math.floor', b)  # 12
            c1 = math.floor(b / 2)
            printt('math.floor', c1)  # 6
            d = b - c1
            printt(d)  # 6

            if index1[0] + c1 > time.shape[1]:
                right = time.shape[1]
            else:
                right = index1[0] + c1

            if index1[0] - d < 0:
                left = 0
            else:
                left = index1[0] - d
            upsampled_prototype_pattern[right:, :] = 0
            upsampled_prototype_pattern[:left, :] = 0
            upsampled_prototype_pattern = upsampled_prototype_pattern.numpy()

            # masked_similarity_map = np.ones(upsampled_prototype_pattern.shape)
            # masked_similarity_map[
            #     upsampled_prototype_pattern < np.max(upsampled_prototype_pattern)] = 0  # < np.max(similarity_map)
            # mask similarity map such that only the nearest patch z* is visualized
            # printt('masked_similarity_map', masked_similarity_map.shape)  # 256,

            # input()

            # 3
            # plt.imsave(
            #     fname=os.path.join('heatmap', prefix, '%s_heatmap_original_image_%s.png' % (str(r_node_id), direction)),
            #     arr=overlayed_original_img, vmin=0.0, vmax=1.0)

            # 3_2

            printt('原始时间序列x', time.shape)  # 3,206

            grayscale_cam = upsampled_prototype_pattern  # similarity_map #rescaled_sim_map  # masked_similarity_map upsampled_prototype_pattern
            c = np.exp(grayscale_cam) / np.sum(np.exp(grayscale_cam))  # *255
            # c=c.transpose(1,0)
            # plt.plot(time[index[2]])
            area = 60 * grayscale_cam
            # np.arange(1,207,1)
            printt('index2',index[2])#2
            printt(time[index[2]].shape)  # (206,)

            x = range(length)
            y1 = np.array(time[index[2]])

            # you
            #
            plt.plot(range(right - 1, time.shape[1]), y1[right - 1:, ], linestyle=':', color='green',
                     linewidth=1)
            # zhong
            printt(len(range(left, right)))

            # 出错x and y must have same first dimension, but have shapes (12,) and (11,)
            # range(196, 208)
            # [0.         0.         0.         0.         0.         0.
            # 0.         0.07265836 0.         0.0531336 ]
            printt('出错', range(left, right))
            printt('出错', y1[left: right])
            plt.plot(range(left, right),
                     y1[left: right], color='orange', linewidth=2)
            # zuo
            plt.plot(range(0, left + 1), y1[:left + 1], linestyle=':', color='green', linewidth=1)

            # plt.show()
            # input()

            sc = plt.scatter(np.arange(length), time[index[2]], c=c, s=area)  # 1536 206 .cpu().detach().numpy()

            # plt.colorbar(sc)
            plt.savefig("D:/桌面/NTree-master/heatmap/{0}/结点{1}_cam_维度{2}.png".format(prefix, str(r_node_id), index[2]))
            # plt.show()
            plt.close()

            # 4
            print('upsampled_prototype_pattern', upsampled_prototype_pattern.shape)  ## 3,206
            rescaled_sim_map1 = rescaled_sim_map.reshape(1, -1)
            print('234', rescaled_sim_map1.shape)
            upsampled_prototype_pattern1 = upsampled_prototype_pattern.reshape(1, -1)
            high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern1,
                                                               0.98)  # rescaled_sim_map upsampled_prototype_pattern
            # high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
            #                  high_act_patch_indices[2]:high_act_patch_indices[3]]#, :

            printt('time', time.shape,index[2])#[2, 640]) 2
            time = time[index[2]:index[2] + 1, :]  # 写错了
            #time = time[index[2]]
            printt('time',time.shape)#[0, 640])

            print('high_act_patch_indices', high_act_patch_indices)
            high_act_patch = time[:, high_act_patch_indices[2]:high_act_patch_indices[3]]  # , :
            # printt('====',high_act_patch.shape)#(1, 12)
            printt('high_act_patch',high_act_patch.shape)#([0, 20])
            # print('time', time.shape)
            printt('plt2',range(high_act_patch_indices[2], high_act_patch_indices[3]))
            plt.plot(range(high_act_patch_indices[2], high_act_patch_indices[3]), high_act_patch[0], color='orange',
                     linewidth=2, marker='o'
                     , markersize='8', markeredgecolor='red')
            plt.savefig(
                "D:/桌面/NTree-master/heatmap/{0}/结点{1}_nearest_patch维度{2}.png".format(prefix, str(r_node_id), index[2]))
            # plt.show()
            plt.close()
            # 4
            # plt.imsave(
            #     fname=os.path.join('heatmap', prefix, '%s_nearest_patch_of_image_%s.png' % (str(r_node_id), direction)),
            #     arr=high_act_patch,
            #     vmin=0.0, vmax=1.0)

    def plot_map2(self, node_id, similarities, prefix, r_node_id, x_np, pool_map, direction, correctness):
        printt('similarities', similarities.shape)
        # similarities = self.linear(similarities)  # [1, 3, 206] ([1, 1, 3, 64])
        # print('similarities', similarities.shape)

        # max = nn.max(similarities)
        # print('max', max)

        for i in range(similarities.shape[2]):
            print('==========================plot map', similarities.shape)
            distance_batch = similarities[0][0][i]  # [1, 1, 768])[1, 3, 768])
            print(distance_batch - 0.37)  # 768,# 206

            # distance_batch = F.relu(distance_batch)
            distance_batch[distance_batch < 0.37 * distance_batch.max()] = distance_batch.min()
            print(distance_batch.shape)
            sz = distance_batch.shape  # 两个参数 ,_
            print('sz', sz)  # [768] [206]
            distance_batch = distance_batch.view(sz, sz, 1)
            print('distance_batch', distance_batch.shape)  # [768] 256

            distance_batch = F.relu(distance_batch)

            similarity_map = distance_batch.detach().cpu().numpy()

            print(similarity_map.shape)  # 206
            # =====归一化
            rescaled_sim_map = similarity_map - np.amin(similarity_map)
            rescaled_sim_map = rescaled_sim_map / np.amax(rescaled_sim_map)
            print('rescaled_sim_map', rescaled_sim_map.shape)  # 206,)
            # ====
            similarity_heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_sim_map), cv2.COLORMAP_JET)
            print('similarity_heatmap1', similarity_heatmap.shape)  # (206, 1, 3)
            similarity_heatmap = np.float32(similarity_heatmap) / 255
            similarity_heatmap = similarity_heatmap[..., ::-1]
            print('similarity_heatmap2', similarity_heatmap.shape)  # (768, 1, 3)  (206, 1, 3)
            # ====
            # 上采样
            # similarity_map=self.upsample1(distance_batch.unsqueeze(dim=0).unsqueeze(dim=0).unsqueeze(dim=0))

            similarity_map = self.upsample2(distance_batch.unsqueeze(dim=0).unsqueeze(dim=0))
            print('upsample sim', similarity_heatmap.shape)

            # 插值
            # similarity_map=distance_batch.unsqueeze(dim=0)
            # print('upsample sim', similarity_map.shape)
            # similarity_map=F.interpolate(similarity_map, scale_factor=2,mode='nearest')
            print('upsample sim', similarity_map)

            similarity_map = similarity_map.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy()
            print('upsample sim', similarity_map.shape)
            masked_similarity_map = np.ones(similarity_map.shape)

            printt(similarity_map.shape)  # (512,)
            printt(np.max(similarity_map))  # 0.3356779
            printt(np.min(similarity_map))  # 0.2767667
            masked_similarity_map[similarity_map < np.max(similarity_map)] = 0  # < np.max(similarity_map)

            # mask similarity map such that only the nearest patch z* is visualized
            print('masked_similarity_map', masked_similarity_map)
            time = x_np.transpose(0, 1)

            upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                     time[0:1, :].shape,
                                                     # dsize=(self.img_size, self.img_size),
                                                     interpolation=cv2.INTER_LANCZOS4)  # INTER_CUBIC INTER_LANCZOS4 INTER_AREA INTER_LINEAR INTER_NEAREST
            print('upsampled_prototype_pattern', upsampled_prototype_pattern)  # (3, 206)
            # =====
            # 1
            # plt.imsave(
            #     fname=os.path.join('heatmap', prefix, '%s_masked_upsampled_heatmap_%s.png' % (str(r_node_id), direction)),
            #     arr=upsampled_prototype_pattern, vmin=0.0, vmax=1.0)

            # similarity_heatmap = cv2.resize(similarity_heatmap,
            #                                 x_np.shape, )
            # dsize=(self.img_size, self.img_size))

            # 2
            # plt.imsave(fname=os.path.join('heatmap', prefix,
            #                               '%s_heatmap_latent_similaritymap_%s.png' % (str(r_node_id), direction)),
            #            arr=similarity_heatmap, vmin=0.0, vmax=1.0)

            print('masked_similarity_map', masked_similarity_map.shape)  # (768, ) 206,
            upsampled_act_pattern = cv2.resize(masked_similarity_map,
                                               x_np.shape,
                                               # disze=(x_np.shape(0), x_np.shape(1)),
                                               # dsize=(self.img_size, self.img_size),
                                               interpolation=cv2.INTER_CUBIC)
            rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
            rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[..., ::-1]  #

            print('heatmap', heatmap.shape)  # #(3, 206, 3) 206 3 3
            # overlayed_original_img = 0.5 * x_np + 0.4 * heatmap

            # 3
            # plt.imsave(
            #     fname=os.path.join('heatmap', prefix, '%s_heatmap_original_image_%s.png' % (str(r_node_id), direction)),
            #     arr=overlayed_original_img, vmin=0.0, vmax=1.0)

            # 3_2

            print('原始时间序列x', time.shape)  # 3,206

            grayscale_cam = upsampled_prototype_pattern  # similarity_map #rescaled_sim_map  # masked_similarity_map upsampled_prototype_pattern
            c = np.exp(grayscale_cam) / np.sum(np.exp(grayscale_cam))  # *255
            # c=c.transpose(1,0)
            plt.plot(time[i])
            area = 60 * grayscale_cam
            # np.arange(1,207,1)
            printt(time[i].shape)
            plt.scatter(np.arange(206), time[i], c=c, s=area)  # 1536 206 .cpu().detach().numpy()
            plt.savefig("D:/桌面/NTree-master/heatmap/{0}/{1}cam_dim{2}.png".format(prefix, str(r_node_id), i))
            plt.show()
            plt.close()

            # 4
            print('upsampled_prototype_pattern', upsampled_prototype_pattern.shape)  ## 3,206
            rescaled_sim_map1 = rescaled_sim_map.reshape(1, -1)
            print('234', rescaled_sim_map1.shape)
            upsampled_prototype_pattern1 = upsampled_prototype_pattern.reshape(1, -1)
            high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern1,
                                                               0.98)  # rescaled_sim_map upsampled_prototype_pattern
            # high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
            #                  high_act_patch_indices[2]:high_act_patch_indices[3]]#, :
            time = time[i:i + 1, :]  # 写错了
            print('high_act_patch_indices', high_act_patch_indices)
            high_act_patch = time[:, high_act_patch_indices[2]:high_act_patch_indices[3]]  # , :
            print(high_act_patch.shape)
            print('time', time.shape)
            plt.plot(range(high_act_patch_indices[2], high_act_patch_indices[3]), high_act_patch[0])
            plt.savefig("D:/桌面/NTree-master/heatmap/{0}/{1}_nearest_patch{2}.png".format(prefix, str(r_node_id), i))
            plt.show()
            plt.close()
            # 4
            # plt.imsave(
            #     fname=os.path.join('heatmap', prefix, '%s_nearest_patch_of_image_%s.png' % (str(r_node_id), direction)),
            #     arr=high_act_patch,
            #     vmin=0.0, vmax=1.0)

    # =========== 5

    # imsave_with_bbox(
    #     fname=os.path.join('heatmap', prefix,
    #                        '%s_bounding_box_nearest_patch_of_image_%s.png' % (str(r_node_id), direction)),
    #     img_rgb=x_np,
    #     bbox_height_start=high_act_patch_indices[0],
    #     bbox_height_end=high_act_patch_indices[1],
    #     bbox_width_start=high_act_patch_indices[2],
    #     bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255) if not correctness else (0, 0, 255))

    def plot_map1(self, node_id, similarities, prefix, r_node_id, x_np, pool_map, direction, correctness):

        similarities = self.linear(similarities)
        print('==========================plot map', similarities.shape)
        distance_batch = similarities[0][0]  # [1, 1, 768])[1, 3, 768])
        print(distance_batch.shape)  # 768,
        distance_batch[distance_batch < distance_batch.max()] = distance_batch.min()
        sz = distance_batch.shape  # 两个参数 ,_
        print('sz', sz)  # [768]
        distance_batch = distance_batch.view(sz, sz, 1)
        print('distance_batch', distance_batch.shape)  # [768]
        similarity_map = distance_batch.detach().cpu().numpy()

        rescaled_sim_map = similarity_map - np.amin(similarity_map)
        rescaled_sim_map = rescaled_sim_map / np.amax(rescaled_sim_map)
        print('rescaled_sim_map', rescaled_sim_map.shape)  # 206,)
        similarity_heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_sim_map), cv2.COLORMAP_JET)
        print('similarity_heatmap1', similarity_heatmap.shape)  # (206, 1, 3)
        similarity_heatmap = np.float32(similarity_heatmap) / 255
        similarity_heatmap = similarity_heatmap[..., ::-1]
        print('similarity_heatmap2', similarity_heatmap.shape)  # (768, 1, 3)  (206, 1, 3)
        masked_similarity_map = np.ones(similarity_map.shape)
        masked_similarity_map[similarity_map < np.max(similarity_map)] = 0
        # mask similarity map such that only the nearest patch z* is visualized

        upsampled_prototype_pattern = cv2.resize(masked_similarity_map,
                                                 x_np.shape,
                                                 # dsize=(self.img_size, self.img_size),
                                                 interpolation=cv2.INTER_CUBIC)

        # 1
        plt.imsave(
            fname=os.path.join('heatmap', prefix, '%s_masked_upsampled_heatmap_%s.png' % (str(r_node_id), direction)),
            arr=upsampled_prototype_pattern, vmin=0.0, vmax=1.0)

        similarity_heatmap = cv2.resize(similarity_heatmap,
                                        x_np.shape, )
        # dsize=(self.img_size, self.img_size))

        # 2
        plt.imsave(fname=os.path.join('heatmap', prefix,
                                      '%s_heatmap_latent_similaritymap_%s.png' % (str(r_node_id), direction)),
                   arr=similarity_heatmap, vmin=0.0, vmax=1.0)

        print('masked_similarity_map', masked_similarity_map.shape)  # (768, ) 206,
        upsampled_act_pattern = cv2.resize(masked_similarity_map,
                                           x_np.shape,
                                           # disze=(x_np.shape(0), x_np.shape(1)),
                                           # dsize=(self.img_size, self.img_size),
                                           interpolation=cv2.INTER_CUBIC)
        rescaled_act_pattern = upsampled_act_pattern - np.amin(upsampled_act_pattern)
        rescaled_act_pattern = rescaled_act_pattern / np.amax(rescaled_act_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255 * rescaled_act_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[..., ::-1]  #

        print('heatmap', heatmap.shape)  # #(3, 206, 3)
        overlayed_original_img = 0.5 * x_np + 0.4 * heatmap

        # 3
        plt.imsave(
            fname=os.path.join('heatmap', prefix, '%s_heatmap_original_image_%s.png' % (str(r_node_id), direction)),
            arr=overlayed_original_img, vmin=0.0, vmax=1.0)

        # 3_2
        plt.close()
        time = x_np.transpose(1, 0)
        print('原始时间序列x', time.shape)  # (206, 3)
        print(masked_similarity_map)
        grayscale_cam = similarity_map  # rescaled_sim_map  # masked_similarity_map
        c = np.exp(grayscale_cam) / np.sum(np.exp(grayscale_cam))
        # c=c.transpose(1,0)
        plt.plot(time[0])
        plt.scatter(np.arange(206), time[0], c=c)  # .cpu().detach().numpy()
        plt.savefig("D:/桌面/NTree-master/heatmap/{0}/cam.png".format(prefix))
        plt.show()
        plt.close()

        print('upsampled_prototype_pattern', upsampled_prototype_pattern.shape)  ## 3,206
        high_act_patch_indices = find_high_activation_crop(upsampled_prototype_pattern, 0.6)

        # high_act_patch = x_np[high_act_patch_indices[0]:high_act_patch_indices[1],
        #                  high_act_patch_indices[2]:high_act_patch_indices[3]]#, :
        high_act_patch = x_np[:,
                         high_act_patch_indices[2]:high_act_patch_indices[3]]  # , :
        print('x_np', x_np.shape)

        # 4
        plt.imsave(
            fname=os.path.join('heatmap', prefix, '%s_nearest_patch_of_image_%s.png' % (str(r_node_id), direction)),
            arr=high_act_patch,
            vmin=0.0, vmax=1.0)

        # 5
        imsave_with_bbox(
            fname=os.path.join('heatmap', prefix,
                               '%s_bounding_box_nearest_patch_of_image_%s.png' % (str(r_node_id), direction)),
            img_rgb=x_np,
            bbox_height_start=high_act_patch_indices[0],
            bbox_height_end=high_act_patch_indices[1],
            bbox_width_start=high_act_patch_indices[2],
            bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255) if not correctness else (0, 0, 255))

    def hard_forward(self, logits, patches, **kwargs):
        batch_size = patches.size(0)
        node_attr = kwargs.setdefault('attr', dict())
        pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=patches.device))

        ps = self.g(**kwargs)
        distance = self._l2_conv(patches, ps, stride=1, dilation=1, padding=0)
        similarity = torch.log((distance + 1) / (distance + 1e-4))

        self.maxim = F.adaptive_max_pool2d(similarity, (1, 1)).squeeze(-1).squeeze(-1)
        to_left = self.maxim[:, 0]
        to_right = 1 - to_left

        if to_left > to_right:
            l_dists, _ = self.l.hard_forward(logits, self.gcb_l(patches), **kwargs)
            return l_dists, node_attr
        else:
            r_dists, _ = self.r.hard_forward(logits, self.gcb_r(patches), **kwargs)

            return r_dists, node_attr

    def explain(self, logits, patches, l_distances, r_distances, x_np, y, prefix, r_node_id, pool_map,
                **kwargs):
        print('================branch explain', patches.shape)  # [1, 768, 51] ([1, 768, 206]) ([1, 192, 5])
        ps = self.g(**kwargs)
        printt('ps', ps.shape)  # ([1, 15, 1])
        distance = self._l2_conv(patches, ps,x_np, stride=1, dilation=1, padding=0,)  # 1, 1, 768] [1, 3, 768])
        similarity = torch.log((distance + 1) / (distance + 1e-4))
        printt('sim', similarity.shape)  # [1, 3, 768])

        # print('maxim = F.adaptive_max_pool1d(similarity, 1)', F.adaptive_max_pool1d(similarity, 1).shape)  # 2 1 1
        maxim = F.adaptive_max_pool2d(similarity, 1).squeeze(-1)  # 原来参数2d (1, 1)
        print('maxim', maxim.shape)  # 2 1 1

        # similarity = F.conv1d(similarity, weight=torch.ones((1, 1, 2, 2)),  # .cuda(), conv2d
        #                       stride=1) if self.proto_size == [1, 1] else similarity

        if self.proto_size == [1, 1]:
            similarity = F.conv1d(similarity, weight=torch.ones((1, 1, 2, 2)),  # .cuda(), conv2d
                                  stride=1)
        else:
            print('123')
            similarity = similarity
        print('sim', similarity.shape)  # 1, 1, 768][1, 3, 768])
        out_map = kwargs['out_map']
        node_id = out_map[self.index]
        self.patch_size = kwargs['img_size']

        to_left = maxim[:, 0]
        to_right = 1 - to_left
        print('left right', to_left,
              to_right)  # tensor([0.2799, 0.2260], grad_fn=<RsubBackward1>) tensor([0.2799, 0.2260], grad_fn=<RsubBackward1>)

        print(f'{node_id} -> ', flush=True)
        print(to_left)
        if to_left > to_right:
            self.plot_map(node_id, similarity, prefix, r_node_id, x_np, pool_map, 'left', True)
            return self.l.explain(logits, self.cbaml(patches), l_distances, r_distances, x_np, y, prefix,
                                  r_node_id * 2 + 1, pool_map,
                                  **kwargs)  # gcb_l cbaml

        else:
            self.plot_map(node_id, similarity, prefix, r_node_id, x_np, pool_map, 'right', True)
            return self.r.explain(logits, self.cbamr(patches), l_distances, r_distances, x_np, y, prefix,
                                  r_node_id * 2 + 2, pool_map,
                                  **kwargs)

    @property
    def size(self) -> int:
        return 1 + self.l.size + self.r.size

    @property
    def leaves(self) -> set:
        return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) -> set:
        return {self}.union(self.l.branches).union(self.r.branches)

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self, **self.l.nodes_by_index, **self.r.nodes_by_index}

    @property
    def num_leaves(self) -> int:
        return self.l.num_leaves + self.r.num_leaves

    @property
    def depth(self) -> int:
        return self.l.depth + 1


def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imshow(img_rgb_float)
    # plt.axis('off')
    plt.imsave(fname, img_rgb_float)


def im_with_bbox(img_rgb, bbox_height_start, bbox_height_end,
                 bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255 * img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end - 1, bbox_height_end - 1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[..., ::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    return img_rgb_float

# def print(*args, **kwargs):
#     flag = False
#     # flag = False
#     if flag:
#         print(*args, **kwargs)
#     else:
#         pass
