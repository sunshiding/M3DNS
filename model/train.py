import torch
from torch import nn
import torch.optim as optim
from torch.autograd import Function
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

import ot
import random
import numpy as np

from loader.load_data import get_batch
  
def pre_train(hp, models, train_data):
    print("----------start pre-training models----------")

    models[1].cuda()
    models[1].train()

    optimizer = optim.Adam([{'params': models[1].parameters()}], lr=hp['pre_lr'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    loss_func = nn.MSELoss()

    for epoch in range(hp['pre_epoch']):
        scheduler.step()
        running_loss = 0.0
        models[1].train()
        batch_size = hp['pre_size']
        data_num = 0
        for i in range(2):
            data = train_data[i]
            bag_num = len(data)
            data_num += bag_num
            max_step = int(bag_num/ batch_size)
            while max_step * batch_size < bag_num:
                max_step += 1

            for step in range(max_step):
                # get data
                step_data = get_batch(data,list(range(step * batch_size,min((step + 1) * batch_size,bag_num))),hp)
                x1, x2, bag1, bag2, y = step_data
                x_text = Variable(x2).cuda()
                b_y = Variable(y).cuda()

                # forward
                h = models[1](x_text,bag2)

                # loss
                loss = loss_func(h, b_y)
                running_loss += loss.data[0] * x[1].size(0)

                # backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # epoch loss
        epoch_loss = running_loss / data_num
        print('epoch {}/{} | Loss: {:.9f}'.format(epoch, hp['pre_epoch'], epoch_loss))
    print("----------end pre-training models----------")
    return models


def train(hp, models, train_data):

    models = pre_train(hp, models,train_data)

    print("----------start training models----------")
    view_num = len(models)  # num of view
    l = hp['label']  # num of label
    # 初始化K0,M矩阵
    k_0 = torch.nn.Softmax()(torch.eye(l))
    k_0 = k_0.data.numpy()
    #k_0_inv = np.linalg.inv(k_0)
    w_loss = WassersteinLoss(k_0)

    trade = hp['trade_off']  # 平衡系数
    lr = hp['lr']
    ae_coe = hp['ae']

    for i in range(view_num):
        models[i].cuda()
    
    par = []
    for i in range(view_num):
        models[i].train()
        par.append({'params': models[i].parameters()})

    optimizer = optim.Adam(par, lr=lr)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    ae_loss = torch.nn.MSELoss(reduce=True, size_average=True)

    def train_for_dataset(data,train_type):
        loss_record = np.zeros(5)
        if data == None:
            return loss_record
        bag_num = len(data)
        data_num += bag_num
        max_step = int(bag_num/ batch_size)
        while max_step * batch_size < bag_num:
            max_step += 1
        for step in range(max_step):
            step_data = get_batch(data,list(range(step * batch_size,min((step + 1) * batch_size,bag_num))),hp)
            x1, x2, bag1, bag2, y = step_data
            if train_type == 0:
                x_img = Variable(x1).cuda()
                x_text = Variable(x2).cuda()
                b_y = Variable(y).cuda()

                # forward
                h1,fea1,dec1 = models[0](x_img,bag1)
                h2,fea2,dec2 = models[1](x_text,bag2)

                # loss
                loss1 = w_loss(h1,by)
                loss2 = w_loss(h2,b_y)
                ae_loss1 = ae_loss(fea1,dec1)
                ae_loss2 = ae_loss(fea2,dec2)

                total_loss = loss1 + loss2 + hp['ae'] * (ae_loss1 + ae_loss2)

                loss_record[0] += loss1.data[0] * x1.size(0)
                loss_record[1] += loss2.data[0] * x1.size(0)
                loss_record[2] += ae_loss1.data[0] * x1.size(0)
                loss_record[3] += ae_loss2.data[0] * x1.size(0)

            elif train_type == 1:
                x_text = Variable(x2).cuda()
                b_y = Variable(y).cuda()

                # forward
                h2,fea2,dec2 = models[1](x_text)

                # loss
                loss2 = w_loss(h2,b_y)
                ae_loss2 = ae_loss(fea2,dec2)

                total_loss = loss2 + hp['ae'] * (ae_loss2)

                loss_record[1] += loss2.data[0] * x2.size(0)
                loss_record[3] += ae_loss2.data[0] * x2.size(0)

            elif train_type == 2:
                x_img = Variable(x1).cuda()
                b_y = Variable(y).cuda()

                # forward
                h1,fea1,dec1 = models[0](x_img)

                # loss
                loss1 = w_loss(h1,b_y)
                ae_loss1 = ae_loss(fea1,dec1)

                total_loss = loss1 + hp['ae'] * (ae_loss1)

                loss_record[0] += loss1.data[0] * x1.size(0)
                loss_record[2] += ae_loss1.data[0] * x1.size(0)

            elif train_type == 3:
                x_img = Variable(x1).cuda()
                x_text = Variable(x2).cuda()

                # forward
                h1,fea1,dec1 = models[0](x_img)
                h2,fea2,dec2 = models[1](x_text)

                # loss
                semi_loss = w_loss(h1,h2)
                ae_loss1 = ae_loss(fea1,dec1)
                ae_loss2 = ae_loss(fea2,dec2)

                total_loss = semi_loss + hp['ae'] * (ae_loss1 + ae_loss2)

                loss_record[2] += ae_loss1.data[0] * x1.size(0)
                loss_record[3] += ae_loss2.data[0] * x1.size(0)
                loss_record[4] += semi_loss.data[0] * x1.size(0)

            elif train_type == 4:
                x_text = Variable(x2).cuda()

                # forward
                h2,fea2,dec2 = models[1](x_text)

                # loss
                ae_loss2 = ae_loss(fea2,dec2)

                total_loss = ae_loss2

                loss_record[3] += ae_loss2.data[0] * x2.size(0)

            elif train_type == 5:
                x_img = Variable(x1).cuda()

                # forward
                h1,fea1,dec1 = models[0](x_img)

                # loss
                ae_loss1 = ae_loss(fea1,dec1)

                total_loss = ae_loss1

                loss_record[2] += ae_loss1.data[0] * x1.size(0)

            # backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        return loss_record

    store_loss = np.zeros((hp['epoch']*hp['epoch_1'],5))
    for epoch in range(hp['epoch']):
        for epoch_1 in range(hp['epoch_1']):
            scheduler.step()
            for i in range(len(train_data)):
                data = train_data[i]
                loss_for_dataset = train_for_dataset(data,i)
                store_loss[epoch * hp['epoch_1'] + epoch_1] = loss_for_dataset.reshape((1,-1))

        # seconde stage
        K = 0
        if not hp['fixed']:
            for i in range(view_num):
                models[i].eval()
            T = np.zeros((l, l))
            m = w_loss.get_m()
            # calculate T
            for i in range(len(train_data)):
                # get data
                data = dataset[i]
                for j in range(len(data)):
                    if j > 2:
                        continue
                    x1, x2, bag1, bag2, b_y = get_batch(data,[j])
                    b_y = b_y.reshape((-1,))
                    b_y[b_y <= 0] = 1e-9
                    b_y = b_y / np.sum(b_y)
                        
                    x_img = None
                    x_text = None
                    if j == 0 or j == 2:
                        x_img = Variable(x1, volatile=True).cuda()
                        h = models[0](x_img,bag1).cpu().data.numpy()
                        h[h <= 0] = 1e-9
                        h = h / np.sum(h)
                        Gs = ot.sinkhorn(h.reshape(-1), b_y.reshape(-1), m / np.max(m), hp['reg'])
                        T += Gs
                    if j == 0 or j == 1:
                        x_text = Variable(x2, volatile=True).cuda()
                        h = models[j](x_text,bag2).cpu().data.numpy()
                        h[h <= 0] = 1e-9
                        h = h / np.sum(h)
                        Gs = ot.sinkhorn(h.reshape(-1), b_y.reshape(-1), m / np.max(m), hp['reg'])
                        T += Gs
            # T /= (bag_num * view_num)

            # calculate K
            G = np.zeros((l, l))
            for i in range(l):
                for j in range(l):
                    if i == j:
                        for k in range(l):
                            if k != i:
                                G[i][j] -= (T[i][k] + T[k][i])
                    else:
                        G[i][j] = 2 * T[i][j]
            # K = np.linalg.inv(k_0_inv - G / trade)
            K = k_0 + G / trade
            K = (K + K.T) / 2
            u, v = np.linalg.eig(K)
            u[u < 0] = 0
            K = np.dot(v, np.dot(np.diag(u), v.T))

            # calculate M
            w_loss.update_cost_matrix(K)

    # 保存loss
    np.save("{}loss.npy".format(hp['rootdir']), store_loss)
    # 保存corr矩阵
    np.save("{}M.npy".format(hp['rootdir']), w_loss.get_m())
    np.save("{}K.npy".format(hp['rootdir']), K)
    #保存model
    save_model(models, hp['rootdir'])
