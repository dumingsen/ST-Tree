import torch
import os
import argparse
from torch.autograd import Variable
from model.DA_Net import DA_Net
from UtilCollection.util import compute_F1_score, exponential_decay, save_result, plot_roc, random_seed
from dataloader.read_UEA import load_UEA, load_xinliu, load_xinliu_kfold, load_data_kfold_dataloader, \
    load_xinliu_kfold_proportion, load_UEA_exp
import time
# UEA datasets
import numpy as np
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
from model.net import gtnet  ##导入包,net_spatial  net_global
# from model.layer import *
from config import get_config
from model_NTree import build_model


from config import printt
from dataloader.read_UEA import load_data_kfold


def print(*args, **kwargs):
    flag = False
    # flag = False
    if flag:
        print(*args, **kwargs)
    else:
        pass


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
length = 1536 * 2
writer = SummaryWriter('runs/exp')
# 实例化这个类，然后我们就得到了Dataset类型的数据，记下来就将这个类传给DataLoader，就可以了。

parser = argparse.ArgumentParser(description='MF-Net for MTSC')
###===========================================模型参数========================================
parser.add_argument('--data', type=str, default='./data/solar_AL.txt',
                    help='location of the data file')  ###数据集
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)  ###？？？
# parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')  ##是否构造图卷积层
parser.add_argument('--buildA_true', type=bool, default=True,
                    help='whether to construct adaptive adjacency matrix')  # 邻接矩阵
parser.add_argument('--gcn_depth', type=int, default=2, help='graph convolution depth')
parser.add_argument('--num_nodes', type=int, default=963, help='number of nodes/variables')  # 对应样本条数
parser.add_argument('--dropout', type=float, default=50, help='dropout rate')
parser.add_argument('--subgraph_size', type=int, default=20, help='k')  ##注意k需要小于num_nodes#20
parser.add_argument('--node_dim', type=int, default=40, help='dim of nodes')
parser.add_argument('--dilation_exponential', type=int, default=2, help='dilation exponential')

parser.add_argument('--conv_channels', type=int, default=16, help='convolution channels')
parser.add_argument('--residual_channels', type=int, default=16, help='residual channels')  ##
parser.add_argument('--skip_channels', type=int, default=32, help='skip channels')
parser.add_argument('--end_channels', type=int, default=64, help='end channels')

parser.add_argument('--in_dim', type=int, default=1, help='inputs dimension')  # 输入维度
parser.add_argument('--seq_in_len', type=int, default=144, help='input sequence length')  ###输入序列长度，1234
parser.add_argument('--seq_out_len', type=int, default=1, help='output sequence length')  #
parser.add_argument('--horizon', type=int, default=3)  ###？？？
parser.add_argument('--layers', type=int, default=5, help='number of layers')

# parser.add_argument('--batch_size',type=int,default=32,help='batch size')###batch
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='weight decay rate')

parser.add_argument('--clip', type=int, default=5, help='clip')

parser.add_argument('--propalpha', type=float, default=0.05, help='prop alpha')
parser.add_argument('--tanhalpha', type=float, default=3, help='tanh alpha')

parser.add_argument('--epochs', type=int, default=5, help='')
parser.add_argument('--num_split', type=int, default=1, help='number of splits for graphs')
parser.add_argument('--step_size', type=int, default=100, help='step_size')
###=========================================实验参数===================================================

parser.add_argument('--model', type=str, default='MF-Net')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--length', type=int, default=8192, help='Embedding length')
parser.add_argument('--writer_path', type=str, default='runs/exp', help='TensorBoard path')
parser.add_argument('--data_path', type=str, default='D:\桌面\多元时间序列数据集\Multivariate_arff')
#D:\桌面\LabelData1
# ./data
#D:\桌面\多元时间序列数据集\Multivariate_arff
parser.add_argument('--seed', type=int, default=1, help='random seed')
# parser.add_argument('--dropout', type=float, default=0.05, help='attention dropout rate')
parser.add_argument('--batch_size', type=int, default= 4)  # 设置batch_size的目的让模型在训练过程中每次选择批量的数据来进行处理
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--cache_path', type=str, default='./cache')
parser.add_argument('--window', type=int, default=64)  # [32,48,64,80,96]
parser.add_argument('--M_name', type=str, default='ME')

args = parser.parse_args()
M_name = args.M_name
writer = SummaryWriter(args.writer_path)  # visualize
random_seed(args.seed)


###=====================================第2部分参数==========================================
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    ###
    parser.add_argument('--cfg', type=str, default='configs/swin_tree_patch4_window14_448_Epilepsy.yaml',
                        metavar="FILE",
                        help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    ####
    parser.add_argument('--cache_path', type=str, default='./cache')
    parser.add_argument('--data_path', type=str, default='G:/桌面/ViT-NeT-main/data', )
    parser.add_argument('--dataset', default='Epilepsy', help='dataset type')  # Epilepsy
    parser.add_argument('--batch_size', default=8, type=int)
    #####
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--img_size', default=448, type=int, help='images input size')
    parser.add_argument('--num_workers', default=4, type=int)
    ###数据集路径
    parser.add_argument('--data-path', type=str, default='G:/桌面/ViT-NeT-main/data',
                        help='path to dataset')  # ../../../data
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")

    parser.add_argument('--amp-opt-level', type=str, default='00', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training parameters
    parser.add_argument('--rank', default=0, type=int, help='')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='tcp://127.0.0.1:2546', help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='')
    parser.add_argument('--seed', default=0, type=int)

    args1, unparsed = parser.parse_known_args()

    config = get_config(args1)  ##

    return args1, config


_, config = parse_option()


#####################=========================================================

def GetDataAndNet(archive_path, archive_name, wa, prob, mask=1):
    train_x, train_y, test_x, test_y, train_loader, test_loader, num_class\
        = load_UEA_exp(archive, args)  ##trainloader,load_UEA

    # get the length and channel of time series
    time_stmp = train_x.shape[2]  # train_loader.__iter__().__next__()[0].shape[2]  # 长度
    printt('时间序列长度', time_stmp)
    in_channel = train_x.shape[1]  # train_loader.__iter__().__next__()[0].shape[1]  # 多元时间序列 next()
    printt('时间序列通道', in_channel)
    # num_class = DealDataset(train_path).num_class()

    # =========================================================

    # net1 = gtnet(args.gcn_true, args.buildA_true, args.gcn_depth, args.num_nodes,
    #               device, dropout=args.dropout, subgraph_size=args.subgraph_size,
    #               node_dim=args.node_dim, dilation_exponential=args.dilation_exponential,
    #               conv_channels=args.conv_channels, residual_channels=args.residual_channels,
    #               skip_channels=args.skip_channels, end_channels=args.end_channels,
    #
    #               seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
    #               layers=args.layers, propalpha=args.propalpha, tanhalpha=args.tanhalpha, layer_norm_affline=False,
    #             #  ,t=time_stmp,#时间序列长度
    #             #   num_classes=num_class,
    #             #   channels=in_channel,
    #             # hidden_dim = (96, 192, 62),
    #             t = time_stmp,  # 长度
    #             down_dim = length,  # length = 1536 * 2，降维维度
    #             hidden_dim = (96,192),##(96, 192, 62)
    #             layers1 = (2, 2, 6, 2),
    #             heads=(3, 6, 12, 24),
    #             channels=in_channel,
    #             num_classes=num_class,
    #             head_dim=32,
    #             window_size=args.window,#768
    #             downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
    #             relative_pos_embedding=True,
    #             wa=wa,
    #             prob=prob,
    #             mask=mask,
    #
    #               )
    net = build_model(config,
                      t=time_stmp,  # 长度
                      down_dim=length,  # length = 1536 * 2，降维维度
                      hidden_dim=(96, 192, 60),  ##(96, 192, 62)
                      layers1=(2, 2, 6, 2),
                      heads=(3, 6, 12, 24),
                      channels=in_channel,
                      num_classes=num_class,
                      head_dim=32,
                      window_size=args.window,  # 768
                      downscaling_factors=(4, 2, 2, 2),  # 代表多长的时间作为一个特征
                      relative_pos_embedding=True,
                      wa=wa,
                      prob=prob,
                      mask=mask, )
    net = net.to(device)
    # ===========================================================
    # ===========================================
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        net = torch.nn.DataParallel(net)  ##多gpu训练
    return train_x, train_y, test_x, test_y, train_loader, test_loader, net, num_class
        #net, num_class
    # return train_loader, test_loader, net, num_class


def test(epoch, test_loader):
    total_pred = torch.tensor([], dtype=torch.int64).to(device)
    total_true = torch.tensor([], dtype=torch.int64).to(device)
    score_list = []
    label_list = []
    total_test_acc = 0
    # for batch_id, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader)):
    for batch_id, (x, y) in enumerate(test_loader):
        y = y.reshape(-1)
        x = torch.unsqueeze(x, dim=1)
        x = Variable(x).float().to(device)
        y = Variable(y).to(device)
        net.eval()
        start_time = time.time()
        # embedding, encoder, output, pred_y = net(x)#需要修改

        pred_y = net(x)
        pred_y = torch.squeeze(pred_y, dim=1)
        inference_time = time.time() - start_time

        _, y_pred = torch.max(pred_y, -1)
        total_test_acc += (y_pred.cpu() == y.cpu()).sum().item()  # 所以我们在求loss或者accuracy时，一般使用item()，而不是直接取它对应的元素x[1,1]。

        total_pred = torch.cat([total_pred, y_pred], dim=0)
        total_true = torch.cat([total_true, y], dim=0)

        test_loss = loss_func(pred_y, y.to(torch.long))  # loss_func = torch.nn.CrossEntropyLoss()

        niter = epoch * test_loader.dataset.__len__() + batch_id  ###
        if niter % 10 == 0:
            # writer = SummaryWriter('runs/exp')
            writer.add_scalar('Test Loss Curve {0}({1})'.format(M_name, length), test_loss.data.item(), niter)

        score_list.extend(pred_y.detach().cpu().numpy())
        label_list.extend(y.cpu().numpy())

    plot_roc(num_class, label_list, score_list, archive, L=length)  ###

    f1_score, precision, recall = compute_F1_score(total_true, total_pred)

    return total_test_acc, f1_score, precision, recall, inference_time, test_loss


def train(optimizer):
    train_time = 0
    max_accuracy = 0
    plot_train_loss = []
    plot_test_loss = []
    plot_train_acc = []
    plot_test_acc = []
    now = time.localtime()
    now1 = time.time()


    for epoch in range(n_epochs):
        ls = []
        s_time = time.time()
        total_train_acc = 0

        # for batch_id,(x,y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        for batch_id, (x, y) in enumerate(train_loader):
            print('训练数据 x', x.shape)  # [2, 1536, 14]
            print('训练数据 y', y.shape)  #
            print('训练数据 x', x)  #
            print('训练数据 y', y)
            y = y.reshape(-1)
            # time_stmp = train_loader.__iter__().next()[0].shape[2]#长度
            # in_channel = train_loader.__iter__().next()[0].shape[1]#多元时间序列
            # torch ALEXNET
            net.train()  ###训练
            optimizer = exponential_decay(optimizer, LEARNING_RATE, global_epoch, 1, 0.90)

            # 添加unsqueeze增加维度
            x = torch.unsqueeze(x, dim=1)

            # Varibale包含三个属性：
            # data：存储了Tensor，是本体的数据
            # grad：保存了data的梯度，本事是个Variable而非Tensor，与data形状一致
            # grad_fn：指向Function对象，用于反向传播的梯度计算之用
            x = Variable(x).float().to(device)
            print('trainx', x.shape)  # [16,144,62]/[16, 1, 144, 62]#[16, 1, 9, 144]

            y = Variable(y).to(device)  # [16]
            print('标签y', y.shape)
            # output 我们需要的 all_sample
            pred_y = net(x)
            print('pred_y', pred_y.shape)  # [16,2]/torch.Size([16, 1, 2])
            # embedding, encoder, output, pred_y = net(x)##模型流入参数

            pred_y = torch.squeeze(pred_y, dim=1)  ###
            print('pred_y', pred_y.shape)

            # loss
            loss = loss_func(pred_y, y.to(torch.long))

            _, y_pred = torch.max(pred_y, -1)
            acc_train = (y_pred.cpu() == y.cpu()).sum().item()
            total_train_acc += acc_train
            niter = epoch * train_loader.dataset.__len__() + batch_id

            if niter % 10 == 0:
                # 功能：将标量添加到 summary
                writer.add_scalar('Train Loss Curve {0}({1})'.format(M_name, length), loss.data.item(), niter)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, y_pred = torch.max(pred_y, -1)
            ls.append(loss)

        printt('训练Epoch: {:04d}'.format(epoch + 1),
               'loss_train: {:.8f}'.format(loss.item()),
               'acc_train: {:.4f}'.format(total_train_acc / train_loader.dataset.__len__()),
               'time: {:.4f}s'.format(time.time() - s_time))

        # np.savetxt("test.csv",total_train_acc / train_loader.dataset.__len__())

        # 训练数据
        plot_train_loss.append(loss.cpu().detach())
        plot_train_acc.append(total_train_acc / train_loader.dataset.__len__())

        # ==================================================================================================####测试开始
        # print("Total time elapsed: {:.4f}s".format(train_time))
        total_test_acc, f1_score, precision, recall, inference_time, test_loss = test(epoch, test_loader)
        plot_test_loss.append(test_loss.cpu().detach())
        plot_test_acc.append(total_test_acc / test_loader.dataset.__len__())

        # save model
        if os.path.exists(f'saved_model/{archive}') == False:  # M_name
            os.makedirs(f'saved_model/{archive}')

        if total_test_acc > max_accuracy:
            print('save best model')
            max_accuracy = total_test_acc
            torch.save(net,
                       f'saved_model/{archive}/{archive} batch={args.batch_size} length={length} window={args.window}.pkl')

        printt('Epoch: {:04d}'.format(epoch + 1),
               'loss_test: {:.8f}'.format(test_loss.item()),
               'acc_test: {:.4f}'.format(total_test_acc / test_loader.dataset.__len__()),
               'time: {:.4f}s'.format(time.time() - s_time))

        # plt.plot()
        train_time = time.time() - now1  # 总运行时间
        # now = time.localtime()
        nowt = time.strftime("%Y-%m-%d-%H_%M_%S", now)  # 实验开始的时间，这一步就是对时间进行格式化
        # print(nowt)

        if os.path.exists(f'acc&loss/{archive}') == False:  # M_name
            os.makedirs(f'acc&loss/{archive}')

        plt.plot(range(len(plot_train_loss)), plot_train_loss, label='train_loss')
        plt.plot(range(len(plot_test_loss)), plot_test_loss, label='test_loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        if os.path.exists(f'acc loss pic') == False:
            os.makedirs(f'acc loss pic')
        plt.savefig('acc&loss/{0}/loss_{1}{2}.png'.format(archive, archive, nowt), bbox_inches='tight')
        # plt.savefig('loss_{0}{1}.png'.format(archive,nowt), bbox_inches='tight')
        # plt.show()
        plt.close()

        plt.plot(range(len(plot_train_acc)), plot_train_acc, label='train_acc')
        plt.plot(range(len(plot_test_acc)), plot_test_acc, label='test_acc')
        plt.xlabel('epoch')
        plt.ylabel('acc')
        plt.legend()
        plt.savefig('acc&loss/{0}/acc_{1}{2}.png'.format(archive, archive, nowt), bbox_inches='tight')
        # plt.savefig('acc_{0}{1}.png'.format(archive,nowt), bbox_inches='tight')
        # plt.show()
        plt.close()

        # file = r'./result/result_{0}.csv'.format(archive)
        if os.path.exists(f'result') == False:
            os.makedirs(f'result')
        save_result(file, ls[-1], total_test_acc / test_loader.dataset.__len__(), f1_score, precision, recall,
                    train_time,
                    inference_time, args.window, length)


wa = 1
prob = 1  ##是否稀疏
if __name__ == '__main__':
    # archives = glob.glob(r'D:/FTP/chengrj/time_series/data/Multivariate_arff/*')
    # for archive_path in archives:
    # archive = os.path.split(archive_path)[-1]

    # ArticularyWordRecognition
    # FaceDetection
    # AtrialFibrillation 单独使用spatial 10epo左右较好
    # BasicMotions 使用了加权拼接
    # CharacterTrajectoriesz最后一层加上了LN
    # Cricket ，开题的模型很合适0.944444
    # DuckDuckGeese
    # EigenWorms
    # Epilepsy
    # HandMovementDirection
    # Heartbeat
    # MotorImagery
    # NATOPS
    # PhonemeSpectra
    # PenDigits
    # SelfRegulationSCP2#50次不够
    # SpokenArabicDigits
    # StandWalkJump 可以跑30次或50以上
    # PEMS-SF
    archive = 'NATOPS'  ##数据集 1234
    #HandMovementDirection SelfRegulationSCP2 StandWalkJump
    # ArticularyWordRecognition  LabelData1 AtrialFibrillation CharacterTrajectories BasicMotions
    # archive = 'PEMS-SF'
    printt(archive)

    file = r'./result/result_{0}.csv'.format(archive)
    # train_loader, test_loader, net, num_class = GetDataAndNet(0, archive, wa, prob)  ##获取数据，网络
    train_x, train_y, test_x, test_y, train_loader, test_loader, net, num_class = GetDataAndNet(0, archive, wa, prob)  ##获取数据，网络

    # for param in net.parameters():
    #     print(param)
    # print(np.sum([np.prod(x.size()) for x in net.parameters()]))

    LEARNING_RATE = 0.001
    optimizer = torch.optim.Adam(
        net.parameters(),
        lr=10,
        betas=(0.9, 0.999),
        eps=1e-08
    )
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    COMPUTE_TRN_METRICS = True
    n_epochs = args.n_epochs

    loss_func = torch.nn.CrossEntropyLoss()

    train(optimizer)  ##训练
    # except:
    #     file = r'./result/result_{0}_{1}_{2}_{3}.csv'.format(str(wa), str(prob), str(mask),
    #                                                          archive)
    #     with open(file, 'a+') as f:
    #         f.write('error\n')
    #     continue
