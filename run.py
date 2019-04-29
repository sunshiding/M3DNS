import os
import numpy as np
from argparse import ArgumentParser
import time
import torch
import random
import pickle
import pdb

from loader.load_data import load_data
from model.model import load_model
from model.model import save_model
from model.train import train
from model.test import test

linear_model = {
    'wzry': [300, 256, 256, 128, 128, 64, 64],
    'flickr': [1277, 1024, 512, 256, 128, 64],
    'iapr': [1312, 1024, 1024, 512, 512, 256, 256, 128, 128, 64],
    'nus': [1000, 512, 512, 256, 256, 128, 128, 64],
    'coco': [2211, 2048, 1024, 1024, 512, 512, 256, 256, 128, 128],
}
label_num = {
    'wzry': 54,
    'flickr': 39,
    'iapr': 10,
    'nus': 10,
    'coco': 20,
}

if __name__ == '__main__':
    parser  = ArgumentParser('M3DN')
    parser.add_argument('--gpu',type=str)
    parser.add_argument('--dataname',type=str)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--pre_epoch',type=int)
    parser.add_argument('--pre_size',type=int)
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--epoch_1',type=int)
    parser.add_argument('--batch_size',type=int)
    parser.add_argument('--label',type=int)
    parser.add_argument('--fixed',type=int)
    parser.add_argument('--pretrain',type=int)
    parser.add_argument('--ratio',type=float)
    parser.add_argument('--train',type=bool,default=True)
    parser.add_argument('--test',type=bool,default=True)
    parser.add_argument('--modelpath',type=str,default="")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seed = args.seed
    torch.manual_seed(seed) # cpu
    torch.cuda.manual_seed(seed) #gpu
    np.random.seed(seed) #numpy
    random.seed(seed) #random and transforms
    torch.backends.cudnn.deterministic = True # cudnn

    hp = {}
    hp['dataname'] = args.dataname
    hp['fixed'] = args.fixed
    hp['pretrain'] = args.pretrain
    hp['ratio'] = args.ratio
    hp['train'] = args.train  # 是否训练模型 
    hp['test'] = args.test  # 是否测试模型
    
    hp['pre_epoch'] = args.pre_epoch  # 预训练轮数
    hp['pre_size'] = args.pre_size  # 预训练batch szie
    hp['pre_lr'] = 0.001  # 预训练初始学习率

    hp['epoch'] = args.epoch  # 训练轮数
    hp['reg'] = 1  # entropic regularization coefficient倒数
    hp['epoch_1'] = args.epoch_1  # 每轮优化,第1阶段迭代次数
    hp['batch_size'] = [args.batch_size] # 网络训练batch size
    hp['lr'] = [0.0001, 0.0001, 0.0001]  # 网络训练初始学习率

    hp['step_size'] = 500  # 学习率衰减步长
    hp['gamma'] = 0.5  # 学习率衰减指数

    hp['trade_off'] = 1  # 平衡系数
    hp['ae'] = 0         # ae loss的系数

    hp['eval'] = [0, 1, 2, 3, 4, 5]  # 评价指标
    hp['thread'] = 0.5  # 预测标签阈值
    
    hp['label'] = args.label
    hp['neure_num'] = linear_model[args.dataname]

    # print("hyper parameter information: ")
    # for key in hp.keys():
    #     print(key, hp[key])

    time_str = time.strftime("%m%d-%H%M",time.localtime(time.time()))
    rootdir = "{}/{}/{}/".format("/data/yangy/data_prepare/result",hp['dataname'],time_str)
    os.makedirs(rootdir, exist_ok=True)
    hp['rootdir'] = rootdir

    pickle.dump(hp, open(rootdir+'parameter.pkl', 'wb'))

    # 获取模型
    my_models = load_model(hp['label'], hp['neure_num'], hp['pretrain'],"{}/{}/{}/".format("./result",hp['dataname'],args.modelpath))

    #获取数据
    train_data, test_data = load_data(hp)

    # 训练模型
    my_models = train(hp, my_models, train_data)

    # 保存模型
    save_model(my_models,rootdir)
    
    # 测试模型
    result = test(test_data, hp, my_models)
    
