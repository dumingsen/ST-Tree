import math
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.io.arff import loadarff
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from config import printt

np.set_printoptions(suppress=True)


def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass


# =======================================================================================
# def load_data_kfold_dataloader(archive_name, args):  # _proportion
#     train_x, train_y, num_class = load_plane_kfold_proportion(archive_name, args)
#
#     # 被试者1
#     train = list(zip(train_x, train_y))
#     printt('数据总长度', len(train))  #
#     train.sort(key=take2)
#     for i in train:
#         print('第二列标签', i[1])
#
#     # =============
#     # 1`
#     value_cnt2 = {}
#     for value in np.array(train_y).reshape(-1):
#         # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
#         value_cnt2[value] = value_cnt2.get(value, 0) + 1
#     # 打印输出结果
#     printt(value_cnt2)  # {3: 253, 1: 90, 4: 75, 2: 260, 0: 12}
#     value_cnt2 = sorted(value_cnt2.items(), key=lambda d: d[0], reverse=False)
#     printt(value_cnt2)  # [(0, 12), (1, 90), (2, 260), (3, 253), (4, 75)]
#
#     # 2
#
#     ##被试者1的数据集 ============================================
#     a0 = []
#     a1 = []
#     a2 = []
#     a3 = []
#     a4 = []
#     p = 0.8
#     for i in train:
#         if i[1] == 0:
#             a0.append(i)
#         elif i[1] == 1:
#             a1.append(i)
#         elif i[1] == 2:
#             a2.append(i)
#         elif i[1] == 3:
#             a3.append(i)
#         elif i[1] == 4:
#             a4.append(i)
#
#     a0_train = a0[0:math.ceil(p * len(a0)):]
#     a0_test = a0[math.ceil(p * len(a0)):]
#     printt('长度a0 a0train a0test', len(a0), len(a0_train), len(a0_test))  # 长度a0 a0train a0test 12 10 2
#
#     a1_train = a1[0:math.ceil(p * len(a1)):]
#     a1_test = a1[math.ceil(p * len(a1)):]
#
#     a2_train = a2[0:math.ceil(p * len(a2)):]
#     a2_test = a2[math.ceil(p * len(a2)):]
#
#     a3_train = a3[0:math.ceil(p * len(a3)):]
#     a3_test = a3[math.ceil(p * len(a3)):]
#     printt('长度a3 a3train a3test', len(a3), len(a3_train), len(a3_test))  # 长度a3 a3train a3test 253 203 50
#
#     a4_train = a4[0:math.ceil(p * len(a4)):]
#     a4_test = a4[math.ceil(p * len(a4)):]
#     a_train = np.concatenate([a0_train, a1_train, a2_train, a3_train, a4_train], axis=0)
#     printt('被试者1的train长度', len(a_train))  # 被试者1的train长度 553
#     a_test = np.concatenate([a0_test, a1_test, a2_test, a3_test, a4_test], axis=0)
#     printt('被试者1的test长度', len(a_test))  # 被试者1的test长度 137
#     # 被试者2的数据集=====================================================
#     a0 = []
#     a1 = []
#     a2 = []
#     a3 = []
#     a4 = []
#     p = 0.8
#     for i in test:
#         if i[1] == 0:
#             a0.append(i)
#         elif i[1] == 1:
#             a1.append(i)
#         elif i[1] == 2:
#             a2.append(i)
#         elif i[1] == 3:
#             a3.append(i)
#         elif i[1] == 4:
#             a4.append(i)
#
#     a0_train = a0[0:math.ceil(p * len(a0)):]
#     a0_test = a0[math.ceil(p * len(a0)):]
#     printt('长度a0 a0train a0test', len(a0), len(a0_train), len(a0_test))  # 长度a0 a0train a0test 14 12 2
#
#     a1_train = a1[0:math.ceil(p * len(a1)):]
#     a1_test = a1[math.ceil(p * len(a1)):]
#
#     a2_train = a2[0:math.ceil(p * len(a2)):]
#     a2_test = a2[math.ceil(p * len(a2)):]
#
#     a3_train = a3[0:math.ceil(p * len(a3)):]
#     a3_test = a3[math.ceil(p * len(a3)):]
#     printt('长度a3 a3train a3test', len(a3), len(a3_train), len(a3_test))  # 长度a3 a3train a3test 291 233 58
#
#     a4_train = a4[0:math.ceil(p * len(a4)):]
#     a4_test = a4[math.ceil(p * len(a4)):]
#     b_train = np.concatenate([a0_train, a1_train, a2_train, a3_train, a4_train], axis=0)
#     printt('被试者2的train长度', len(b_train))  # 被试者2的train长度 555
#     b_test = np.concatenate([a0_test, a1_test, a2_test, a3_test, a4_test], axis=0)
#     printt('被试者2的train长度', len(b_test))  # 被试者1的train长度 135
#     ##只使用单个被试者1==============================
#     a = 2
#     if a == 1:
#         train_x = []
#         train_y = []
#         test_x = []
#         test_y = []
#         for i in a_train:
#             train_x.append(i[0])
#             train_y.append(i[1])
#
#         for i in a_test:
#             test_x.append(i[0])
#             test_y.append(i[1])
#     ##只使用单个被试者2==============================
#     elif a == 2:
#         train_x = []
#         train_y = []
#         test_x = []
#         test_y = []
#         for i in b_train:
#             train_x.append(i[0])
#             train_y.append(i[1])
#
#         for i in b_test:
#             test_x.append(i[0])
#             test_y.append(i[1])
#     # 被试者1和2划分好的数据集==========================
#     # input()
#     elif a == 3:
#         train = np.concatenate([a_train, b_train], axis=0)
#         test = np.concatenate([a_test, b_test], axis=0)
#         printt('训练集的长度', len(train))  # 1108
#         printt('测试机的长度', len(test), )  # 272
#         train_x = []
#         train_y = []
#         test_x = []
#         test_y = []
#         for i in train:
#             train_x.append(i[0])
#             train_y.append(i[1])
#
#         for i in test:
#             test_x.append(i[0])
#             test_y.append(i[1])
#     # ==================================
#     train_x = np.array(train_x)
#     train_y = np.array(train_y)
#     test_x = np.array(test_x)
#     test_y = np.array(test_y)
#     ##======================
#     # input()
#     TrainDataset = DealDataset(train_x, train_y)
#     TestDataset = DealDataset(test_x, test_y)
#
#     train_loader = DataLoader(dataset=TrainDataset,
#                               batch_size=args.batch_size,
#                               shuffle=True)
#     test_loader = DataLoader(dataset=TestDataset,
#                              batch_size=args.batch_size,
#                              shuffle=True)
#     return train_loader, test_loader

def load_plane_kfold_dataloader(archive_name, args):  # _proportion
    train_x, train_y, num_class = load_plane_kfold_proportion(archive_name, args)

    # 被试者1
    train = list(zip(train_x, train_y))
    printt('数据总长度', len(train))  #
    train.sort(key=take2)
    for i in train:
        print('第二列标签', i[1])

    # =============
    # 1`
    value_cnt2 = {}
    for value in np.array(train_y).reshape(-1):
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        value_cnt2[value] = value_cnt2.get(value, 0) + 1
    # 打印输出结果
    printt(value_cnt2)  # {3: 253, 1: 90, 4: 75, 2: 260, 0: 12}
    value_cnt2 = sorted(value_cnt2.items(), key=lambda d: d[0], reverse=False)
    printt(value_cnt2)  # [(0, 12), (1, 90), (2, 260), (3, 253), (4, 75)]

    # 2

    ##被试者1的数据集 ============================================
    a0 = []
    a1 = []

    p = 0.7
    for i in train:
        if i[1] == 0:
            a0.append(i)
        elif i[1] == 1:
            a1.append(i)

    #类别1
    a0_train = a0[0:math.ceil(p * len(a0)):]
    a0_test = a0[math.ceil(p * len(a0)):]
    printt('长度a0 a0train a0test', len(a0), len(a0_train), len(a0_test))  # 长度a0 a0train a0test 12 10 2

    #类别2
    a1_train = a1[0:math.ceil(p * len(a1)):]
    a1_test = a1[math.ceil(p * len(a1)):]

    #训练和测试集
    a_train = np.concatenate([a0_train, a1_train], axis=0)
    printt('飞机的train长度', len(a_train))  # 被试者1的train长度 553
    a_test = np.concatenate([a0_test, a1_test], axis=0)
    printt('飞机的test长度', len(a_test))  # 被试者1的test长度 137
    # 被试者2的数据集=====================================================

    ##只使用单个被试者1==============================
    a = 1
    if a == 1:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in a_train:
            train_x.append(i[0])
            train_y.append(i[1])

        for i in a_test:
            test_x.append(i[0])
            test_y.append(i[1])
    ##只使用单个被试者2==============================

    # 被试者1和2划分好的数据集==========================
    # input()

    # ==================================
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    ##======================
    # input()
    TrainDataset = DealDataset(train_x, train_y)
    TestDataset = DealDataset(test_x, test_y)

    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)
    return train_loader, test_loader


# ==========================================================================

def load_plane_kfold_proportion(archive_name, args):
    cache_path = f'{args.cache_path}/{archive_name}_proportion.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        # train_x, train_y, num_class = torch.load(cache_path)
        train_x, train_y, num_class = torch.load(cache_path)

    else:

        path = args.data_path  # 待读取的文件夹
        filename = '{0}\{1}.txt'.format(path, archive_name)
        data = np.loadtxt(filename, dtype=np.float32, delimiter=',')
        printt('飞机数据', np.array(data).shape)  # 飞机数据 (500, 7)

        # input()
        data = np.array(data)

        # data1 = np.reshape(data, (-1, data.shape[1] * 100))
        # printt('飞机数据', data1.shape)  # (5, 700)

        data = np.reshape(data, (-1, 100, data.shape[1],))
        printt('飞机数据', data.shape)  # (5, 7, 100)

        label = []
        for i in data:
            # printt('i', i.shape)
            a = i[:, -1]
            label.append(a[0])
        printt('标签', label)

        data = data.transpose(0, 2, 1)

        data = data[:, :-1]
        # printt(data[0])
        printt('data', data.shape)
        label = np.array(label)
        train_y = label
        train_x = data
        # input()

        value_cnt1 = {}
        for value in np.array(label).reshape(-1):
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt1[value] = value_cnt1.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt1)  # {3: 544, 1: 189, 4: 147, 2: 474, 0: 26}

        # ==================================================================================
        input()
        # 放到0 - Numclass
        labels = np.unique(train_y)  ##标签
        printt('labels', labels)  # [0 1 2 3 4]
        num_class = len(labels)
        printt('num_class', num_class)  # 5

        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        # test_y = np.vectorize(transform.get)(test_y)
        printt('train_x', train_x.shape)  # (47, 14, 1536)
        # printt('test_x', test_x.shape)  # (47 , 14, 1536)
        printt('train_y', train_y.shape)  # (47, 14, 1536)
        # printt('test_y', test_y.shape)  # (47 , 14, 1536)
        input()
        torch.save((train_x, train_y, num_class), cache_path)

    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)
    #
    # train_loader = DataLoader(dataset=TrainDataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=True)
    # test_loader = DataLoader(dataset=TestDataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=True)

    return train_x, train_y, num_class
    # return train_x, train_y, test_x, test_y, train_loader, test_loader, num_class
    # return train_x, train_y, num_class


# ========================================================================================

#
def take2(elem):
    return elem[1]


##for 普通版本
def load_data_kfold_dataloader(archive_name, args):  # _proportion
    train_x, train_y, test_x, test_y, num_class = load_xinliu_kfold_proportion(archive_name, args)

    # 被试者1
    train = list(zip(train_x, train_y))
    printt('数据总长度', len(train))  #
    train.sort(key=take2)
    for i in train:
        print('第二列标签', i[1])
    # 被试者2
    test = list(zip(test_x, test_y))
    printt('数据总长度', len(test))  #
    test.sort(key=take2)
    # =============
    # 1`
    value_cnt2 = {}
    for value in np.array(train_y).reshape(-1):
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        value_cnt2[value] = value_cnt2.get(value, 0) + 1
    # 打印输出结果
    printt(value_cnt2)  # {3: 253, 1: 90, 4: 75, 2: 260, 0: 12}
    value_cnt2 = sorted(value_cnt2.items(), key=lambda d: d[0], reverse=False)
    printt(value_cnt2)  # [(0, 12), (1, 90), (2, 260), (3, 253), (4, 75)]

    # 2
    value_cnt3 = {}
    for value in np.array(test_y).reshape(-1):
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        value_cnt3[value] = value_cnt3.get(value, 0) + 1
    # 打印输出结果
    printt(value_cnt3)  # { 3: 291,1: 99, 4: 72,2: 214,  0: 14}
    value_cnt3 = sorted(value_cnt3.items(), key=lambda d: d[0], reverse=False)
    printt(value_cnt3)  # [(0, 14), (1, 99), (2, 214), (3, 291), (4, 72)]
    ##被试者1的数据集 ============================================
    a0 = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    p = 0.8
    for i in train:
        if i[1] == 0:
            a0.append(i)
        elif i[1] == 1:
            a1.append(i)
        elif i[1] == 2:
            a2.append(i)
        elif i[1] == 3:
            a3.append(i)
        elif i[1] == 4:
            a4.append(i)

    a0_train = a0[0:math.ceil(p * len(a0)):]
    a0_test = a0[math.ceil(p * len(a0)):]
    printt('长度a0 a0train a0test', len(a0), len(a0_train), len(a0_test))  # 长度a0 a0train a0test 12 10 2

    a1_train = a1[0:math.ceil(p * len(a1)):]
    a1_test = a1[math.ceil(p * len(a1)):]

    a2_train = a2[0:math.ceil(p * len(a2)):]
    a2_test = a2[math.ceil(p * len(a2)):]

    a3_train = a3[0:math.ceil(p * len(a3)):]
    a3_test = a3[math.ceil(p * len(a3)):]
    printt('长度a3 a3train a3test', len(a3), len(a3_train), len(a3_test))  # 长度a3 a3train a3test 253 203 50

    a4_train = a4[0:math.ceil(p * len(a4)):]
    a4_test = a4[math.ceil(p * len(a4)):]
    a_train = np.concatenate([a0_train, a1_train, a2_train, a3_train, a4_train], axis=0)
    printt('被试者1的train长度', len(a_train))  # 被试者1的train长度 553
    a_test = np.concatenate([a0_test, a1_test, a2_test, a3_test, a4_test], axis=0)
    printt('被试者1的test长度', len(a_test))  # 被试者1的test长度 137
    # 被试者2的数据集=====================================================
    a0 = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    p = 0.8
    for i in test:
        if i[1] == 0:
            a0.append(i)
        elif i[1] == 1:
            a1.append(i)
        elif i[1] == 2:
            a2.append(i)
        elif i[1] == 3:
            a3.append(i)
        elif i[1] == 4:
            a4.append(i)

    a0_train = a0[0:math.ceil(p * len(a0)):]
    a0_test = a0[math.ceil(p * len(a0)):]
    printt('长度a0 a0train a0test', len(a0), len(a0_train), len(a0_test))  # 长度a0 a0train a0test 14 12 2

    a1_train = a1[0:math.ceil(p * len(a1)):]
    a1_test = a1[math.ceil(p * len(a1)):]

    a2_train = a2[0:math.ceil(p * len(a2)):]
    a2_test = a2[math.ceil(p * len(a2)):]

    a3_train = a3[0:math.ceil(p * len(a3)):]
    a3_test = a3[math.ceil(p * len(a3)):]
    printt('长度a3 a3train a3test', len(a3), len(a3_train), len(a3_test))  # 长度a3 a3train a3test 291 233 58

    a4_train = a4[0:math.ceil(p * len(a4)):]
    a4_test = a4[math.ceil(p * len(a4)):]
    b_train = np.concatenate([a0_train, a1_train, a2_train, a3_train, a4_train], axis=0)
    printt('被试者2的train长度', len(b_train))  # 被试者2的train长度 555
    b_test = np.concatenate([a0_test, a1_test, a2_test, a3_test, a4_test], axis=0)
    printt('被试者2的train长度', len(b_test))  # 被试者1的train长度 135
    ##只使用单个被试者1==============================
    a = 2
    if a == 1:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in a_train:
            train_x.append(i[0])
            train_y.append(i[1])

        for i in a_test:
            test_x.append(i[0])
            test_y.append(i[1])
    ##只使用单个被试者2==============================
    elif a == 2:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in b_train:
            train_x.append(i[0])
            train_y.append(i[1])

        for i in b_test:
            test_x.append(i[0])
            test_y.append(i[1])
    # 被试者1和2划分好的数据集==========================
    # input()
    elif a == 3:
        train = np.concatenate([a_train, b_train], axis=0)
        test = np.concatenate([a_test, b_test], axis=0)
        printt('训练集的长度', len(train))  # 1108
        printt('测试机的长度', len(test), )  # 272
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in train:
            train_x.append(i[0])
            train_y.append(i[1])

        for i in test:
            test_x.append(i[0])
            test_y.append(i[1])
    # ==================================
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    ##======================
    # input()
    TrainDataset = DealDataset(train_x, train_y)
    TestDataset = DealDataset(test_x, test_y)

    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)
    return train_loader, test_loader


##划分比例, for mtpool版本
def load_data_kfold(archive_name, args):  # _proportion
    train_x, train_y, test_x, test_y, num_class = load_xinliu_kfold_proportion(archive_name, args)

    # 被试者1
    train = list(zip(train_x, train_y))
    printt('数据总长度', len(train))  #
    train.sort(key=take2)
    for i in train:
        print('第二列标签', i[1])
    # 被试者2
    test = list(zip(test_x, test_y))
    printt('数据总长度', len(test))  #
    test.sort(key=take2)
    # =============
    # 1`
    value_cnt2 = {}
    for value in np.array(train_y).reshape(-1):
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        value_cnt2[value] = value_cnt2.get(value, 0) + 1
    # 打印输出结果
    printt(value_cnt2)  # {3: 253, 1: 90, 4: 75, 2: 260, 0: 12}
    value_cnt2 = sorted(value_cnt2.items(), key=lambda d: d[0], reverse=False)
    printt(value_cnt2)  # [(0, 12), (1, 90), (2, 260), (3, 253), (4, 75)]

    # 2
    value_cnt3 = {}
    for value in np.array(test_y).reshape(-1):
        # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
        value_cnt3[value] = value_cnt3.get(value, 0) + 1
    # 打印输出结果
    printt(value_cnt3)  # { 3: 291,1: 99, 4: 72,2: 214,  0: 14}
    value_cnt3 = sorted(value_cnt3.items(), key=lambda d: d[0], reverse=False)
    printt(value_cnt3)  # [(0, 14), (1, 99), (2, 214), (3, 291), (4, 72)]
    ##被试者1的数据集 ============================================
    a0 = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    p = 0.8
    for i in train:
        if i[1] == 0:
            a0.append(i)
        elif i[1] == 1:
            a1.append(i)
        elif i[1] == 2:
            a2.append(i)
        elif i[1] == 3:
            a3.append(i)
        elif i[1] == 4:
            a4.append(i)

    a0_train = a0[0:math.ceil(p * len(a0)):]
    a0_test = a0[math.ceil(p * len(a0)):]
    printt('长度a0 a0train a0test', len(a0), len(a0_train), len(a0_test))  # 长度a0 a0train a0test 12 10 2

    a1_train = a1[0:math.ceil(p * len(a1)):]
    a1_test = a1[math.ceil(p * len(a1)):]

    a2_train = a2[0:math.ceil(p * len(a2)):]
    a2_test = a2[math.ceil(p * len(a2)):]

    a3_train = a3[0:math.ceil(p * len(a3)):]
    a3_test = a3[math.ceil(p * len(a3)):]
    printt('长度a3 a3train a3test', len(a3), len(a3_train), len(a3_test))  # 长度a3 a3train a3test 253 203 50

    a4_train = a4[0:math.ceil(p * len(a4)):]
    a4_test = a4[math.ceil(p * len(a4)):]
    a_train = np.concatenate([a0_train, a1_train, a2_train, a3_train, a4_train], axis=0)
    printt('被试者1的train长度', len(a_train))  # 被试者1的train长度 553
    a_test = np.concatenate([a0_test, a1_test, a2_test, a3_test, a4_test], axis=0)
    printt('被试者1的test长度', len(a_test))  # 被试者1的test长度 137
    # 被试者2的数据集=====================================================
    a0 = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    p = 0.8
    for i in test:
        if i[1] == 0:
            a0.append(i)
        elif i[1] == 1:
            a1.append(i)
        elif i[1] == 2:
            a2.append(i)
        elif i[1] == 3:
            a3.append(i)
        elif i[1] == 4:
            a4.append(i)

    a0_train = a0[0:math.ceil(p * len(a0)):]
    a0_test = a0[math.ceil(p * len(a0)):]
    printt('长度a0 a0train a0test', len(a0), len(a0_train), len(a0_test))  # 长度a0 a0train a0test 14 12 2

    a1_train = a1[0:math.ceil(p * len(a1)):]
    a1_test = a1[math.ceil(p * len(a1)):]

    a2_train = a2[0:math.ceil(p * len(a2)):]
    a2_test = a2[math.ceil(p * len(a2)):]

    a3_train = a3[0:math.ceil(p * len(a3)):]
    a3_test = a3[math.ceil(p * len(a3)):]
    printt('长度a3 a3train a3test', len(a3), len(a3_train), len(a3_test))  # 长度a3 a3train a3test 291 233 58

    a4_train = a4[0:math.ceil(p * len(a4)):]
    a4_test = a4[math.ceil(p * len(a4)):]
    b_train = np.concatenate([a0_train, a1_train, a2_train, a3_train, a4_train], axis=0)
    printt('被试者2的train长度', len(b_train))  # 被试者2的train长度 555
    b_test = np.concatenate([a0_test, a1_test, a2_test, a3_test, a4_test], axis=0)
    printt('被试者2的test长度', len(b_test))  # 被试者2的train长度 135
    ##只使用单个被试者1==============================
    a = 1
    if a == 1:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in a_train:
            train_x.append(i[0])
            train_y.append(i[1])

        for i in a_test:
            test_x.append(i[0])
            test_y.append(i[1])
    ##只使用单个被试者2==============================
    elif a == 2:
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in b_train:
            train_x.append(i[0])
            train_y.append(i[1])

        for i in b_test:
            test_x.append(i[0])
            test_y.append(i[1])
    # 被试者1和2划分好的数据集==========================
    # input()
    elif a == 3:
        train = np.concatenate([a_train, b_train], axis=0)
        test = np.concatenate([a_test, b_test], axis=0)
        printt('训练集的长度', len(train))  # 1108
        printt('测试机的长度', len(test), )  # 272
        train_x = []
        train_y = []
        test_x = []
        test_y = []
        for i in train:
            train_x.append(i[0])
            train_y.append(i[1])

        for i in test:
            test_x.append(i[0])
            test_y.append(i[1])
    # 去噪函数使用==================================
    # train_x = denoise(train_x)
    # test_x = denoise(test_x)

    # ===============================只使用频率
    train_x = fft(train_x)
    test_x = fft(test_x)

    # ===================================
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)
    ##======================
    # 归一化===============================
    scaler = StandardScaler()
    scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
    train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
    test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
    # printt(train_x.shape)  # (553, 14, 1536)
    # input()
    return train_x, train_y, test_x, test_y, num_class


from scipy.signal import filtfilt, iirnotch, freqz, butter


def filter_50(signal):
    fs = 500  # 500.0  # Sample frequency (Hz)
    f0 = 50  # Frequency to be removed from signal (Hz)
    w0 = f0 / (fs / 2)  # Normalized Frequency
    Q = 30
    b, a = iirnotch(w0, Q)
    signal = filtfilt(b, a, signal)
    return (signal)


import matplotlib.pyplot as plt


def denoise(data):
    # printt('data',np.array(data).shape)
    a = []
    for sample in data:
        for ts in sample:
            ts = ts.reshape(-1)
            # printt('before', np.array(ts).shape)  # (1536,)
            b = filter_50(ts)
            # printt('after', np.array(b).shape)  # (1536,)
            a.append(b)
            # printt('a', np.array(a).shape)
            # input()

    a = np.array(a)
    a = a.reshape(np.array(data).shape)
    printt('降噪后的数据形状', a.shape)  # (553, 14, 1536)
    plt.plot(np.array(data)[0][1], label='origin')
    plt.plot(a[0][1], label='denoise')
    # plt.show()
    # input()
    return a


def fft(data):
    # printt('data',np.array(data).shape)
    a = []
    N = np.array(data).shape[2]
    for sample in data:
        for ts in sample:
            ts = ts.reshape(-1)
            # printt('before', np.array(ts).shape)  # (1536,)
            b = np.abs(np.fft.rfft(ts) / N)  # fast fourier transform
            # printt('after', np.array(b).shape)  # (1536,) 769
            b = np.pad(b, (0, N - 769), 'constant', constant_values=(0))
            a.append(b)
            # printt('a', np.array(a).shape)
            # input()

    a = np.array(a)
    a = a.reshape(np.array(data).shape)
    printt('fft后的数据形状', a.shape)  # (553, 14, 1536)
    # plt.plot(np.array(data)[0][0], label='origin')
    plt.plot(a[0][0], label='denoise')
    # plt.show()
    # input()
    return a


##获得训练数据，被试者1作为训练，2作为训练，
def load_data_kfold_diftest(batch_size, k, n, archive_name, args):
    _, _, _, _, train, test, num_class = load_xinliu_kfold_diftest(archive_name, args)
    return train, test


# k折交叉
def load_data_kfold1(batch_size, k, n, archive_name, args):
    # This function using functions in preprocessing.py to build dataset,
    # and then randomly split dataset with a fixed random seed.
    printt("Building DataSet ...")

    # create_path()
    # data = build_dataset()  # Build data set
    data_xinliu, label, num_class = load_xinliu_kfold(archive_name, args)
    printt('data xinliou', data_xinliu.shape)  # (1380, 14, 1536)
    printt('label xinliou', label.shape)  # (1380,)

    # data = [data_xinliu, label.reshape(-1,1)]
    # for a, b in zip(data_xinliu, label)
    data = list(zip(data_xinliu, label))
    printt('数据总长度', len(data))  # 1380
    # for i in range(len(data)):
    #     printt('zip kfold', data[i][0].shape, data[i][1])  # (1536, 14) (1,)

    # np.set_printoptions(threshold=np.inf)
    # f = open(".\datalabel.txt", "w")
    # f.writelines(str(data))
    # f.close()

    print("Complete.")

    printt("Splitting DataSet ...")

    l = len(data)
    print(l)
    shuffle_dataset = True
    random_seed = 42  # fixed random seed
    indices = list(range(l))

    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)  # shuffle

    # Collect indexes of samples for validation set.
    val_indices = indices[int(l / k) * n:int(l / k) * (n + 1)]
    # Collect indexes of samples for train set. Here the logic is that a sample
    # cannot in train set if already in validation set
    train_indices = list(set(indices).difference(set(val_indices)))
    printt('train index', train_indices)
    printt('val index', val_indices)  #

    printt('train index数量', len(train_indices))
    printt('val index数量', len(val_indices))
    # sampler
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)  # build Sampler
    valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    #
    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)

    # loader
    train_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                               sampler=train_sampler
                                               , )  # build dataloader for train set
    validation_loader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                                    sampler=valid_sampler,
                                                    )  # build dataloader for validate set
    printt("Complete.")
    return train_loader, validation_loader


# 读取原始心流数据专用=========================================================================
def load_xinliu_kfold_old(archive_name, args):
    cache_path = f'{args.cache_path}/{archive_name}_kfold.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, num_class = torch.load(cache_path)
        # train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)

    else:
        test1_data = []
        test2_data = []
        test1_label = []
        test2_label = []
        path = args.data_path  # 待读取的文件夹
        path_list = os.listdir(path)
        path_list.sort()  # 对读取的路径进行排序
        print(path_list)
        for filename in path_list:
            print(os.path.join(path, filename))  # path为路径，可以去掉，只显示文件名
            path1 = os.path.join(path, filename)
            path_list1 = os.listdir(path1)
            path_list1.sort()  # 对读取的路径进行排序
            print(path_list1)

            # 1 ================分别获取受试者1和受试者2的地址
            path_test1 = os.path.join(path1, path_list1[0])
            print('path_test1', path_test1)  # D:\桌面\LabelData1\016\016-1
            path_test2 = os.path.join(path1, path_list1[1])

            # 2.1 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test1)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test1, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test1, label_test1 = extract_xinliu(path_times_list1)

                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            test1_data.append(data_test1)
            test1_label.append(label_test1)
            # test1_data = np.array(test1_data)
            # test1_label = np.array(test1_label)
            print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)

            # 2.2 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test2)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test2, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test2, label_test2 = extract_xinliu(path_times_list1)

                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            test2_data.append(data_test2)
            test2_label.append((label_test2))
            # test2_data = np.array(test2_data)
            # test2_label = np.array(test2_label)
            print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)

        # 完整的初始数据
        test1_data = np.array(test1_data)
        test1_label = np.array(test1_label)
        test2_data = np.array(test2_data)
        test2_label = np.array(test2_label)
        printt('被试者1_data', test1_data.shape)  # (47, 1, 14, 1536)
        printt('被试者2_data', test2_data.shape)  # (47, 1, 14, 1536)
        printt('被试者1_label', test1_label.shape)  # (47, 1)
        printt('被试者2_label', test2_label.shape)  # (47, 1)
        test1_data = test1_data.squeeze(1)
        test2_data = test2_data.squeeze(1)
        printt(test1_data.shape)

        train_x = test1_data  # .transpose(0, 2, 1)
        test_x = test2_data  # .transpose(0, 2, 1)
        train_y = test1_label
        test_y = test2_label

        # xinliu_data = train_x.append(test_x)
        xinliu_data = np.concatenate((train_x, test_x), axis=0)
        printt('pinjie', xinliu_data.shape)  # (94, 1536, 14)
        # xinliu_label = train_y.append(test_y)
        xinliu_label = np.concatenate((train_y, test_y), axis=0)

        # 异常值填充
        # train_x, train_y = extract_data(train_data)  ##y为标签
        # test_x, test_y = extract_data(test_data)
        # train_x[np.isnan(train_x)] = 0
        # test_x[np.isnan(test_x)] = 0

        # 归一化数据
        # scaler = StandardScaler()
        # scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        # train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        # test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
        # printt('train_x', train_x.shape)  # (47, 14, 1536)
        # printt('test_x', test_x.shape)  # (47, 14, 1536)
        scaler = StandardScaler()
        scaler.fit(xinliu_data.reshape(-1, xinliu_data.shape[-1]))
        train_x = scaler.transform(xinliu_data.reshape(-1, xinliu_data.shape[-1])).reshape(xinliu_data.shape)
        printt('xinliudata', train_x.shape)  # (47, 14, 1536)

        # 放到0-Numclass
        labels = np.unique(xinliu_label)  ##标签
        printt('labels', labels)  # [0 1 2 3 4]
        num_class = len(labels)
        printt('num_class', num_class)  # 5

        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(xinliu_label)
        # test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, num_class), cache_path)

    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)
    # # return TrainDataset,TestDataset,len(labels)
    # # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # # batchszie：批大小，决定一个epoch有多少个Iteration；
    # train_loader = DataLoader(dataset=TrainDataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=True)
    # test_loader = DataLoader(dataset=TestDataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=True)

    # return train_loader, test_loader, num_class
    return train_x, train_y, num_class


# proportion,按每个被试者的类别的比例划分数据集专用
def load_xinliu_kfold_proportion(archive_name, args):
    cache_path = f'{args.cache_path}/{archive_name}_proportion.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        # train_x, train_y, num_class = torch.load(cache_path)
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)

    else:
        test1_data = []
        test2_data = []
        test_data = []

        test1_label = []
        test2_label = []
        test_label = []
        path = args.data_path  # 待读取的文件夹
        path_list = os.listdir(path)
        path_list.sort()  # 对读取的路径进行排序
        print(path_list)
        for filename in path_list:
            print(os.path.join(path, filename))  # path为路径，可以去掉，只显示文件名
            path1 = os.path.join(path, filename)
            path_list1 = os.listdir(path1)
            path_list1.sort()  # 对读取的路径进行排序
            print(path_list1)

            # 1 分别获取受试者1和受试者2的地址
            path_test1 = os.path.join(path1, path_list1[0])
            print('path_test1', path_test1)  # D:\桌面\LabelData1\016\016-1
            path_test2 = os.path.join(path1, path_list1[1])

            # ===================================================================================2.1 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test1)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test1, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test1, label_test1 = extract_xinliu(path_times_list1)
                            print('cut data', data_test1.shape)

                            test1_data.append(data_test1)
                            test1_label.append(label_test1)
                            # test1_data = np.array(test1_data)
                            # test1_label = np.array(test1_label)
                            print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)
                            # 按文件夹顺序
                            test_data.append(data_test1)
                            test_label.append(label_test1)
                            # input()
                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            # test1_data.append(data_test1)
            # test1_label.append(label_test1)
            # # test1_data = np.array(test1_data)
            # # test1_label = np.array(test1_label)
            # print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)
            # # 按文件夹顺序
            # test_data.append(data_test1)
            # test_label.append(label_test1)

            # ====================================================================================2.2 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test2)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test2, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test2, label_test2 = extract_xinliu(path_times_list1)
                            # printt('cut data', data_test2.shape)
                            test2_data.append(data_test2)
                            test2_label.append((label_test2))
                            # test2_data = np.array(test2_data)
                            # test2_label = np.array(test2_label)
                            print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)
                            # 总的
                            test_data.append(data_test2)
                            test_label.append(label_test2)

                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            # test2_data.append(data_test2)
            # test2_label.append((label_test2))
            # # test2_data = np.array(test2_data)
            # # test2_label = np.array(test2_label)
            # print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)
            # # 总的
            # test_data.append(data_test2)
            # test_label.append(label_test2)

        # 总的完整数据
        # test_data = np.array(test_data)
        # test_label = np.array(test_label)
        # # test_data = test_data.squeeze(1)
        # # printt('test data', test_data.shape) #(276, 5, 14, 1536)
        # # printt('test label', test_label.shape) #(276, 5)
        # test_data = np.reshape(test_data, (-1, test_data.shape[2], test_data.shape[3]))
        # test_label = np.reshape(test_label, (-1, 1))
        # printt('test data', test_data.shape)  #
        # printt('test label', test_label.shape)  #
        # input()

        # # 分被试者的完整的初始数据
        test1_data = np.array(test1_data)
        test1_label = np.array(test1_label)
        test2_data = np.array(test2_data)
        test2_label = np.array(test2_label)

        test1_data = np.reshape(test1_data, (-1, test1_data.shape[2], test1_data.shape[3]))
        test1_label = np.reshape(test1_label, (-1, 1))
        test2_data = np.reshape(test2_data, (-1, test2_data.shape[2], test2_data.shape[3]))
        test2_label = np.reshape(test2_label, (-1, 1))

        printt('被试者1_data', test1_data.shape)  # (690, 14, 1536)
        printt('被试者2_data', test2_data.shape)  # (690, 1)
        printt('被试者1_label', test1_label.shape)  #
        printt('被试者2_label', test2_label.shape)  #
        # test1_data = test1_data.squeeze(1)
        # test2_data = test2_data.squeeze(1)
        # printt(test1_data.shape)
        #
        train_x = test1_data  # .transpose(0, 2, 1)
        test_x = test2_data  # .transpose(0, 2, 1)
        train_y = test1_label
        test_y = test2_label
        #
        # # xinliu_data = train_x.append(test_x)
        # xinliu_data = np.concatenate((train_x, test_x), axis=0)
        # printt('pinjie', xinliu_data.shape)  # (94, 1536, 14)
        # # xinliu_label = train_y.append(test_y)
        # xinliu_label = np.concatenate((train_y, test_y), axis=0)

        # 异常值填充
        # train_x, train_y = extract_data(train_data)  ##y为标签
        # test_x, test_y = extract_data(test_data)

        # train_x[np.isnan(train_x)] = 0
        # test_x[np.isnan(test_x)] = 0

        # =======================================归一化数据模块
        # scaler = StandardScaler()
        # scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        # train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        # test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
        # printt('train_x', train_x.shape)  # (47, 14, 1536)
        # printt('test_x', test_x.shape)  # (47 , 14, 1536)

        # scaler = StandardScaler()
        # scaler.fit(test_data.reshape(-1, test_data.shape[-1]))
        # train_x = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
        # printt('xinliudata', train_x.shape)  # (47, 14, 1536)

        # ===============================================

        value_cnt1 = {}
        for value in np.array(test_label).reshape(-1):
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt1[value] = value_cnt1.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt1)  # {3: 544, 1: 189, 4: 147, 2: 474, 0: 26}

        value_cnt2 = {}
        for value in np.array(test1_label).reshape(-1):
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt2[value] = value_cnt2.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt2)  # {3: 253, 1: 90, 4: 75, 2: 260, 0: 12}

        value_cnt3 = {}
        for value in np.array(test2_label).reshape(-1):
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt3[value] = value_cnt3.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt3)  # { 3: 291,1: 99, 4: 72,2: 214,  0: 14}

        # ==================================================================================
        input()
        # 放到0 - Numclass
        labels = np.unique(train_y)  ##标签
        printt('labels', labels)  # [0 1 2 3 4]
        num_class = len(labels)
        printt('num_class', num_class)  # 5

        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)
        printt('train_x', train_x.shape)  # (47, 14, 1536)
        printt('test_x', test_x.shape)  # (47 , 14, 1536)
        printt('train_y', train_y.shape)  # (47, 14, 1536)
        printt('test_y', test_y.shape)  # (47 , 14, 1536)
        input()
        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)
    #
    # train_loader = DataLoader(dataset=TrainDataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=True)
    # test_loader = DataLoader(dataset=TestDataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=True)

    return train_x, train_y, test_x, test_y, num_class
    # return train_x, train_y, test_x, test_y, train_loader, test_loader, num_class
    # return train_x, train_y, num_class


# 被试者1作为训练
def load_xinliu_kfold_diftest(archive_name, args):
    cache_path = f'{args.cache_path}/{archive_name}_diftest.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        # train_x, train_y, num_class = torch.load(cache_path)
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)

    else:
        test1_data = []
        test2_data = []
        test_data = []

        test1_label = []
        test2_label = []
        test_label = []
        path = args.data_path  # 待读取的文件夹
        path_list = os.listdir(path)
        path_list.sort()  # 对读取的路径进行排序
        print(path_list)
        for filename in path_list:
            print(os.path.join(path, filename))  # path为路径，可以去掉，只显示文件名
            path1 = os.path.join(path, filename)
            path_list1 = os.listdir(path1)
            path_list1.sort()  # 对读取的路径进行排序
            print(path_list1)

            # 1 分别获取受试者1和受试者2的地址
            path_test1 = os.path.join(path1, path_list1[0])
            print('path_test1', path_test1)  # D:\桌面\LabelData1\016\016-1
            path_test2 = os.path.join(path1, path_list1[1])

            # ===================================================================================2.1 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test1)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test1, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test1, label_test1 = extract_xinliu(path_times_list1)
                            print('cut data', data_test1.shape)

                            test1_data.append(data_test1)
                            test1_label.append(label_test1)
                            # test1_data = np.array(test1_data)
                            # test1_label = np.array(test1_label)
                            print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)
                            # 按文件夹顺序
                            test_data.append(data_test1)
                            test_label.append(label_test1)
                            # input()
                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            # test1_data.append(data_test1)
            # test1_label.append(label_test1)
            # # test1_data = np.array(test1_data)
            # # test1_label = np.array(test1_label)
            # print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)
            # # 按文件夹顺序
            # test_data.append(data_test1)
            # test_label.append(label_test1)

            # ====================================================================================2.2 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test2)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test2, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test2, label_test2 = extract_xinliu(path_times_list1)
                            # printt('cut data', data_test2.shape)
                            test2_data.append(data_test2)
                            test2_label.append((label_test2))
                            # test2_data = np.array(test2_data)
                            # test2_label = np.array(test2_label)
                            print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)
                            # 总的
                            test_data.append(data_test2)
                            test_label.append(label_test2)

                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            # test2_data.append(data_test2)
            # test2_label.append((label_test2))
            # # test2_data = np.array(test2_data)
            # # test2_label = np.array(test2_label)
            # print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)
            # # 总的
            # test_data.append(data_test2)
            # test_label.append(label_test2)

        # 总的完整数据
        # test_data = np.array(test_data)
        # test_label = np.array(test_label)
        # # test_data = test_data.squeeze(1)
        # # printt('test data', test_data.shape) #(276, 5, 14, 1536)
        # # printt('test label', test_label.shape) #(276, 5)
        # test_data = np.reshape(test_data, (-1, test_data.shape[2], test_data.shape[3]))
        # test_label = np.reshape(test_label, (-1, 1))
        # printt('test data', test_data.shape)  #
        # printt('test label', test_label.shape)  #
        # input()

        # # 分被试者的完整的初始数据
        test1_data = np.array(test1_data)
        test1_label = np.array(test1_label)
        test2_data = np.array(test2_data)
        test2_label = np.array(test2_label)

        test1_data = np.reshape(test1_data, (-1, test1_data.shape[2], test1_data.shape[3]))
        test1_label = np.reshape(test1_label, (-1, 1))
        test2_data = np.reshape(test2_data, (-1, test2_data.shape[2], test2_data.shape[3]))
        test2_label = np.reshape(test2_label, (-1, 1))

        printt('被试者1_data', test1_data.shape)  # (690, 14, 1536)
        printt('被试者2_data', test2_data.shape)  # (690, 1)
        printt('被试者1_label', test1_label.shape)  #
        printt('被试者2_label', test2_label.shape)  #
        # test1_data = test1_data.squeeze(1)
        # test2_data = test2_data.squeeze(1)
        # printt(test1_data.shape)
        #
        train_x = test1_data  # .transpose(0, 2, 1)
        test_x = test2_data  # .transpose(0, 2, 1)
        train_y = test1_label
        test_y = test2_label
        #
        # # xinliu_data = train_x.append(test_x)
        # xinliu_data = np.concatenate((train_x, test_x), axis=0)
        # printt('pinjie', xinliu_data.shape)  # (94, 1536, 14)
        # # xinliu_label = train_y.append(test_y)
        # xinliu_label = np.concatenate((train_y, test_y), axis=0)

        # 异常值填充
        # train_x, train_y = extract_data(train_data)  ##y为标签
        # test_x, test_y = extract_data(test_data)

        # train_x[np.isnan(train_x)] = 0
        # test_x[np.isnan(test_x)] = 0

        # 归一化数据
        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
        printt('train_x', train_x.shape)  # (47, 14, 1536)
        printt('test_x', test_x.shape)  # (47 , 14, 1536)

        # scaler = StandardScaler()
        # scaler.fit(test_data.reshape(-1, test_data.shape[-1]))
        # train_x = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
        # printt('xinliudata', train_x.shape)  # (47, 14, 1536)

        # ===============================================

        value_cnt1 = {}
        for value in np.array(test_label).reshape(-1):
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt1[value] = value_cnt1.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt1)  # {3: 544, 1: 189, 4: 147, 2: 474, 0: 26}

        value_cnt2 = {}
        for value in np.array(test1_label).reshape(-1):
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt2[value] = value_cnt2.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt2)  # {3: 253, 1: 90, 4: 75, 2: 260, 0: 12}

        value_cnt3 = {}
        for value in np.array(test2_label).reshape(-1):
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt3[value] = value_cnt3.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt3)  # { 3: 291,1: 99, 4: 72,2: 214,  0: 14}

        # ==================================================================================
        input()
        # 放到0 - Numclass
        labels = np.unique(train_y)  ##标签
        printt('labels', labels)  # [0 1 2 3 4]
        num_class = len(labels)
        printt('num_class', num_class)  # 5

        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)
        printt('train_x', train_x.shape)  # (47, 14, 1536)
        printt('test_x', test_x.shape)  # (47 , 14, 1536)
        printt('train_y', train_y.shape)  # (47, 14, 1536)
        printt('test_y', test_y.shape)  # (47 , 14, 1536)
        input()
        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    TrainDataset = DealDataset(train_x, train_y)
    TestDataset = DealDataset(test_x, test_y)
    # # return TrainDataset,TestDataset,len(labels)
    # # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # # batchszie：批大小，决定一个epoch有多少个Iteration；
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    return train_x, train_y, test_x, test_y, train_loader, test_loader, num_class
    # return train_x, train_y, num_class


###k折
def load_xinliu_kfold(archive_name, args):
    cache_path = f'{args.cache_path}/{archive_name}_kfold.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, num_class = torch.load(cache_path)
        # train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)

    else:
        test1_data = []
        test2_data = []
        test_data = []

        test1_label = []
        test2_label = []
        test_label = []
        path = args.data_path  # 待读取的文件夹
        path_list = os.listdir(path)
        path_list.sort()  # 对读取的路径进行排序
        print('path_list', path_list)
        for filename in path_list:
            print(os.path.join(path, filename))  # path为路径，可以去掉，只显示文件名
            path1 = os.path.join(path, filename)
            path_list1 = os.listdir(path1)
            path_list1.sort()  # 对读取的路径进行排序
            print('path_list1', path_list1)  # ['016-1', '016-2']

            # 1 分别获取受试者1和受试者2的地址
            path_test1 = os.path.join(path1, path_list1[0])
            print('path_test1', path_test1)  # D:\桌面\LabelData1\016\016-1
            path_test2 = os.path.join(path1, path_list1[1])

            # ===================================================================================2.1 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test1)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            # path_test22 = os.listdir(path_test2)
            # path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test1, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    # '016-1-1_EPOCPLUS_171243_2022.11.20T11.24.51+08.00.mc.pm.fe.bp.csv', '016-1-1_label.csv', '016-1-1_label_delcol.csv', '_ChangeTime.csv', '_Options.csv', 'cutData']
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test1, label_test1 = extract_xinliu(path_times_list1)
                            print('cut data', data_test1.shape)

                            test1_data.append(data_test1)
                            test1_label.append(label_test1)
                            # test1_data = np.array(test1_data)
                            # test1_label = np.array(test1_label)
                            print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)
                            # 按文件夹顺序
                            test_data.append(data_test1)
                            test_label.append(label_test1)
                            # input()
                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            # test1_data.append(data_test1)
            # test1_label.append(label_test1)
            # # test1_data = np.array(test1_data)
            # # test1_label = np.array(test1_label)
            # print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)
            # # 按文件夹顺序
            # test_data.append(data_test1)
            # test_label.append(label_test1)

            # ====================================================================================2.2 每个受试者的轮数访问，第几次玩游戏
            # path_test11 = os.listdir(path_test2)
            # path_test11.sort()
            # printt('path_test11',
            #       path_test11)  # ['016-2-0_EPOCPLUS_171243_2022.11.20T11.21.19+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            print('path_test22',
                  path_test22)
            for t1 in path_test22:
                path_times = os.path.join(path_test2, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    # path_times_list ['016-2-1_EPOCPLUS_171243_2022.11.20T11.24.52+08.00.mc.pm.fe.bp.csv', '016-2-1_label.csv', '016-2-1_label_delcol.csv', '_ChangeTime.csv', '_Options.csv', 'cutData']
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test2, label_test2 = extract_xinliu(path_times_list1)
                            # printt('cut data', data_test2.shape)
                            test2_data.append(data_test2)
                            test2_label.append((label_test2))
                            # test2_data = np.array(test2_data)
                            # test2_label = np.array(test2_label)
                            print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)
                            # 总的
                            test_data.append(data_test2)
                            test_label.append(label_test2)

                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            # test2_data.append(data_test2)
            # test2_label.append((label_test2))
            # # test2_data = np.array(test2_data)
            # # test2_label = np.array(test2_label)
            # print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)
            # # 总的
            # test_data.append(data_test2)
            # test_label.append(label_test2)

        # 总的完整数据
        test_data = np.array(test_data)
        test_label = np.array(test_label)
        # np.set_printoptions(threshold = np.inf)
        # f = open(".\label1.txt", "w")
        # f.writelines(str(test_label))
        # f.close()
        print(test_label)
        # test_data = test_data.squeeze(1)
        # printt('test data', test_data.shape) #(276, 5, 14, 1536)
        # printt('test label', test_label.shape) #(276, 5)
        test_data = np.reshape(test_data, (-1, test_data.shape[2], test_data.shape[3]))
        test_label = np.reshape(test_label, (-1))
        # np.set_printoptions(threshold = np.inf)
        # f = open(".\label2.txt", "w")
        # f.writelines(str(test_label))
        # f.close()
        print('test data', test_data.shape)  #
        print('test label', test_label.shape)  #
        print(test_label)
        # input()

        # # 分被试者的完整的初始数据
        # test1_data = np.array(test1_data)
        # test1_label = np.array(test1_label)
        # test2_data = np.array(test2_data)
        # test2_label = np.array(test2_label)
        # printt('被试者1_data', test1_data.shape)  # (47, 1, 14, 1536)
        # printt('被试者2_data', test2_data.shape)  # (47, 1, 14, 1536)
        # printt('被试者1_label', test1_label.shape)  # (47, 1)
        # printt('被试者2_label', test2_label.shape)  # (47, 1)
        # test1_data = test1_data.squeeze(1)
        # test2_data = test2_data.squeeze(1)
        # printt(test1_data.shape)
        #
        # train_x = test1_data#.transpose(0, 2, 1)
        # test_x = test2_data#.transpose(0, 2, 1)
        # train_y = test1_label
        # test_y = test2_label
        #
        # # xinliu_data = train_x.append(test_x)
        # xinliu_data = np.concatenate((train_x, test_x), axis=0)
        # printt('pinjie', xinliu_data.shape)  # (94, 1536, 14)
        # # xinliu_label = train_y.append(test_y)
        # xinliu_label = np.concatenate((train_y, test_y), axis=0)

        # 异常值填充
        # train_x, train_y = extract_data(train_data)  ##y为标签
        # test_x, test_y = extract_data(test_data)
        # train_x[np.isnan(train_x)] = 0
        # test_x[np.isnan(test_x)] = 0

        # 归一化数据
        # scaler = StandardScaler()
        # scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        # train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        # test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
        # printt('train_x', train_x.shape)  # (47, 14, 1536)
        # printt('test_x', test_x.shape)  # (47 , 14, 1536)
        scaler = StandardScaler()
        scaler.fit(test_data.reshape(-1, test_data.shape[-1]))
        train_x = scaler.transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)
        printt('xinliudata', train_x.shape)  # (1380, 14, 1536)

        # 放到0 - Numclass
        value_cnt = {}
        for value in test_label:
            # get(value, num)函数的作用是获取字典中value对应的键值, num=0指示初始值大小。
            value_cnt[value] = value_cnt.get(value, 0) + 1
        # 打印输出结果
        printt(value_cnt)  # {3: 544, 1: 189, 4: 147, 2: 474, 0: 26}
        labels = np.unique(test_label)  ##标签
        printt('labels', labels)  # [0 1 2 3 4]
        num_class = len(labels)
        printt('num_class', num_class)  # 5

        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(test_label)
        # test_y = np.vectorize(transform.get)(test_y)
        input()
        torch.save((train_x, train_y, num_class), cache_path)

    # TrainDataset = DealDataset(train_x, train_y)
    # TestDataset = DealDataset(test_x, test_y)
    # # return TrainDataset,TestDataset,len(labels)
    # # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # # batchszie：批大小，决定一个epoch有多少个Iteration；
    # train_loader = DataLoader(dataset=TrainDataset,
    #                           batch_size=args.batch_size,
    #                           shuffle=True)
    # test_loader = DataLoader(dataset=TestDataset,
    #                          batch_size=args.batch_size,
    #                          shuffle=True)

    # return train_loader, test_loader, num_class
    return train_x, train_y, num_class


##
def load_xinliu(archive_name, args):
    cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)

    else:
        test1_data = []
        test2_data = []
        test1_label = []
        test2_label = []
        path = args.data_path  # 待读取的文件夹
        path_list = os.listdir(path)
        path_list.sort()  # 对读取的路径进行排序
        print(path_list)
        for filename in path_list:
            print(os.path.join(path, filename))  # path为路径，可以去掉，只显示文件名
            path1 = os.path.join(path, filename)
            path_list1 = os.listdir(path1)
            path_list1.sort()  # 对读取的路径进行排序
            print(path_list1)

            # 1 ================分别获取受试者1和受试者2的地址
            path_test1 = os.path.join(path1, path_list1[0])
            print('path_test1', path_test1)  # D:\桌面\LabelData1\016\016-1
            path_test2 = os.path.join(path1, path_list1[1])

            # 2.1 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test1)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test1, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test1, label_test1 = extract_xinliu(path_times_list1)

                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            test1_data.append(data_test1)
            test1_label.append(label_test1)
            # test1_data = np.array(test1_data)
            # test1_label = np.array(test1_label)
            print('被试者1', np.array(test1_data).shape)  # (1, 1, 14, 1536)

            # 2.2 每个受试者的轮数访问，第几次玩游戏
            path_test11 = os.listdir(path_test2)
            path_test11.sort()
            print('path_test11',
                  path_test11)  # ['016-1-0_EPOCPLUS_171243_2022.11.20T11.12.48+08.00.mc.pm.fe.bp.csv', '1', '2', '3']
            path_test22 = os.listdir(path_test2)
            path_test22.sort()
            for t1 in path_test11:
                path_times = os.path.join(path_test2, t1)
                print('path_times', path_times)
                if not (os.path.isfile(path_times)):
                    print('是文件夹')

                    # 每个轮数下的数据切片获取
                    path_times_list = os.listdir(path_times)
                    path_times_list.sort()
                    print('path_times_list', path_times_list)
                    for times in path_times_list:
                        path_times_list1 = os.path.join(path_times, times)
                        print('path_times_list1', path_times_list1)
                        if not (os.path.isfile(path_times_list1)):

                            # 数据切片的数据获取
                            data_test2, label_test2 = extract_xinliu(path_times_list1)

                        else:
                            continue
                else:
                    print('不是文件夹')
                    continue
            test2_data.append(data_test2)
            test2_label.append((label_test2))
            # test2_data = np.array(test2_data)
            # test2_label = np.array(test2_label)
            print('被试者2', np.array(test2_data).shape)  # (1, 1, 14, 1536)

        # 完整的初始数据
        test1_data = np.array(test1_data)
        test1_label = np.array(test1_label)
        test2_data = np.array(test2_data)
        test2_label = np.array(test2_label)
        printt('被试者1_data', test1_data.shape)  # (47, 1, 14, 1536)
        printt('被试者2_data', test2_data.shape)  # (47, 1, 14, 1536)
        printt('被试者1_label', test1_label.shape)  # (47, 1)
        printt('被试者2_label', test2_label.shape)  # (47, 1)
        test1_data = test1_data.squeeze(1)
        test2_data = test2_data.squeeze(1)
        printt(test1_data.shape)
        train_x = test1_data.transpose(0, 2, 1)
        test_x = test2_data.transpose(0, 2, 1)
        train_y = test1_label
        test_y = test2_label

        # train_x, train_y = extract_data(train_data)  ##y为标签
        # test_x, test_y = extract_data(test_data)
        # train_x[np.isnan(train_x)] = 0
        # test_x[np.isnan(test_x)] = 0

        # 归一化数据
        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)
        printt('train_x', train_x.shape)  # (47, 14, 1536)
        printt('test_x', test_x.shape)  # (47, 14, 1536)

        # 放到0-Numclass
        labels = np.unique(train_y)  ##标签
        printt('labels', labels)  # [0 1 2 3 4]
        num_class = len(labels)
        printt('num_class', num_class)  # 5

        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    TrainDataset = DealDataset(train_x, train_y)
    TestDataset = DealDataset(test_x, test_y)
    # return TrainDataset,TestDataset,len(labels)
    # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # batchszie：批大小，决定一个epoch有多少个Iteration；
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    return train_loader, test_loader, num_class


def extract_xinliu(data):
    a = []
    b = []
    path_list = os.listdir(data)
    path_list.sort()
    print('xinliu csv', path_list)  # ['01.csv', '02.csv', '03.csv', '04.csv', '05.csv']
    for filename in path_list:
        print(os.path.join(data, filename))  # path为路径，可以去掉，只显示文件名
        path = os.path.join(data, filename)
        # printt(path)
        # 读取初始数据
        train_data_csv = pd.read_csv(path, encoding="utf-8")

        # 删除一二列获取的数据和标签
        train_data = train_data_csv.iloc[0:, 2:-1]
        print('train', np.array(train_data).shape)  # (1536, 14)
        train_data = np.array(train_data)
        # print(train_data)
        train_data = train_data.T
        ##print('转置', train_data)
        a.append(train_data)
        # a = np.array(a)
        # print(a.shape)  # (1, 14, 1536)

        # 标签
        train_label = train_data_csv.iloc[:, -1]
        train_label = np.array(train_label)
        # print(train_label)
        print(train_label.shape)
        b.append(train_label[0])
        # b = np.array(b)
        # print(b.shape)
        # break
    a = np.array(a)
    b = np.array(b)
    # printt(a.shape)
    # input()
    return a, b

    # res_data = []
    # res_labels = []
    # for t_data, t_label in data:
    #     t_data = np.array([d.tolist() for d in t_data])
    #     t_label = t_label.decode("utf-8")
    #     res_data.append(t_data)
    #     res_labels.append(t_label)
    # return np.array(res_data).swapaxes(1, 2), np.array(res_labels)


# =========================================================================
def extract_data(data):
    res_data = []
    res_labels = []
    for t_data, t_label in data:
        t_data = np.array([d.tolist() for d in t_data])
        t_label = t_label.decode("utf-8")
        res_data.append(t_data)
        res_labels.append(t_label)
    return np.array(res_data).swapaxes(1, 2), np.array(res_labels)
    # swapaxes的用法就是交换轴的位置，前后两个的位置没有关系。


def load_UEA(archive_name, args):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    # load from cache
    cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)


    # load from arff
    else:
        train_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)  ##y为标签
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)  ##标签
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    TrainDataset = DealDataset(train_x, train_y)
    TestDataset = DealDataset(test_x, test_y)
    # return TrainDataset,TestDataset,len(labels)
    # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # batchszie：批大小，决定一个epoch有多少个Iteration；
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=True)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    return train_loader, test_loader, num_class


def load_UEA_exp(archive_name, args):
    # train_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TRAIN.arff','r',encoding='UTF-8'))[0]
    # test_data = loadarff(open(f'D:/FTP/chengrj/time_series/data/Multivariate_arff/{dataset}/{dataset}_TEST.arff','r',encoding='UTF-8'))[0]

    # load from cache
    cache_path = f'{args.cache_path}/{archive_name}.dat'  ##需要建一个
    if os.path.exists(cache_path) is True:
        print('load form cache....')
        train_x, train_y, test_x, test_y, num_class = torch.load(cache_path)


    # load from arff
    else:
        train_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TRAIN.arff', 'r', encoding='UTF-8'))[0]
        test_data = \
            loadarff(open(f'{args.data_path}/{archive_name}/{archive_name}_TEST.arff', 'r', encoding='UTF-8'))[0]

        train_x, train_y = extract_data(train_data)  ##y为标签
        test_x, test_y = extract_data(test_data)
        train_x[np.isnan(train_x)] = 0
        test_x[np.isnan(test_x)] = 0

        scaler = StandardScaler()
        scaler.fit(train_x.reshape(-1, train_x.shape[-1]))
        train_x = scaler.transform(train_x.reshape(-1, train_x.shape[-1])).reshape(train_x.shape)
        test_x = scaler.transform(test_x.reshape(-1, test_x.shape[-1])).reshape(test_x.shape)

        # 放到0-Numclass
        labels = np.unique(train_y)  ##标签
        num_class = len(labels)
        # print(num_class)
        transform = {k: i for i, k in enumerate(labels)}
        train_y = np.vectorize(transform.get)(train_y)
        test_y = np.vectorize(transform.get)(test_y)

        torch.save((train_x, train_y, test_x, test_y, num_class), cache_path)

    TrainDataset = DealDataset(train_x, train_y)
    TestDataset = DealDataset(test_x, test_y)
    # return TrainDataset,TestDataset,len(labels)
    # DataLoader是Pytorch中用来处理模型输入数据的一个工具类。组合了数据集（dataset） + 采样器(sampler)，
    # 并在数据集上提供单线程或多线程(num_workers )的可迭代对象
    # dataset (Dataset) – 决定数据从哪读取或者从何读取；
    # batchszie：批大小，决定一个epoch有多少个Iteration；
    train_loader = DataLoader(dataset=TrainDataset,
                              batch_size=args.batch_size,
                              shuffle=False)
    test_loader = DataLoader(dataset=TestDataset,
                             batch_size=args.batch_size,
                             shuffle=False)

    return train_x, train_y, test_x, test_y, train_loader, test_loader, num_class


class DealDataset(Dataset):
    """
        下载数据、初始化数据，都可以在这里完成
    """

    def __init__(self, x, y):
        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)
        self.len = x.shape[0]
        # self.x_data = self.x_data.transpose(2, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

    def num_class(self):
        return len(set(self.y_data))


if __name__ == '__main__':
    load_UEA('Ering')
