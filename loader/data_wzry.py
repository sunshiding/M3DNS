import os
import pickle
import torch
import numpy as np
import random

def load_data(path,hp):
    #path = '/data/yangy/data_prepare/wzry/'
    data_num = 116
    test_num = int(116 * 0.3)
    semi_num = int(116 * 0.21)
    train_num = data_num - test_num - semi_num

    data_id = list(range(data_num))
    random.shuffle(data_id)
    train_id = data_id[:train_num]
    semi_id = data_id[train_num:train_num+semi_num]
    test_id = data_id[train_num+semi_num:]

    np.save("{}train_id.npy",train_id)
    np.save("{}semi_id.npy",semi_id)
    np.save("{}test_id.npy",test_id)

    print("----------start reading train data----------")
    train_data = []
    for i in range(len(train_id)):
        pkl_id = train_id[i]
        filepath = path + 'wzry' + str(pkl_id) + '.pkl'
        if os.path.exists(filepath) is True:
            data_temp = pickle.load(open(filepath, 'rb'))
            for key in data_temp.keys():
                if test_is_valid(data_temp[key]):
                    train_data.append(data_temp[key])
    semi_data = []
    for i in range(len(semi_id)):
        pkl_id = semi_id[i]
        filepath = path + 'wzry' + str(pkl_id) + '.pkl'
        if os.path.exists(filepath) is True:
            data_temp = pickle.load(open(filepath, 'rb'))
            for key in data_temp.keys():
                if test_is_valid(data_temp[key]):
                    semi_data.append(data_temp[key])
    test_data = []
    for i in range(len(test_id)):
        pkl_id = test_id[i]
        filepath = path + 'wzry' + str(pkl_id) + '.pkl'
        if os.path.exists(filepath) is True:
            data_temp = pickle.load(open(filepath, 'rb'))
            for key in data_temp.keys():
                if test_is_valid(data_temp[key]):
                    test_data.append(data_temp[key])

    print("train data: ", len(train_data))
    print("vali data: ", len(vali_data))
    print("test data: ", len(test_data))
    text_label_data, img_label_data, text_semi_data, img_semi_data = None, None, None, None
    if ratio == 0:
        return [train_data,text_label_data, img_label_data, semi_data,text_semi_data, img_semi_data], test_data
    else:
        all_label_data, text_label_data, img_label_data = get_imcomplete_data(train_data,ratio)
        all_semi_data, text_semi_data, img_semi_data = get_imcomplete_data(semi_data,ratio)
    return [all_label_data, text_label_data, img_label_data, all_semi_data, text_semi_data, img_semi_data], test_data

def get_imcomplete_data(data,ratio):
    ratio = ratio / 2
    num = len(data)
    imcomplete_num = int(num * ratio)
    data_id = list(range(len(num)))
    random.shuffle(data_id)
    img_id = data_id[:imcomplete_num]
    text_id = data_id[imcomplete_num:imcomplete_num*2]
    all_id = [imcomplete_num*2:]

    all_data = [data[i] for i in all_id]
    img_data = [data[i] for i in img_id]
    text_data = [data[i] for i in text_id]
    return all_data, img_data, text_data

def get_batch(data, indices):
    view_data = [[] for i in range(view_num)]
    bag_size = [[] for i in range(view_num)]
    bag_label = [[] for i in range(view_num)]
    view_num = 2
    for i in indices:
        for j in range(view_num):
            instance = torch.FloatTensor(data[i][j].astype(float))
            if j == 1:
                instance = instance.view(-1,300)
            else:
                instance = instance.view(-1,3,224,224)
            bag_size[j].append(instance.shape[0])
            view_data[j].append(instance)
            bag_label[j].append(torch.FloatTensor(data[i][-1].astype(float).reshape((1,-1))))
    for i in range(view_num):
        if len(view_data[i]) > 0:
            view_data[i] = torch.cat(view_data[i])
            bag_label[i] = torch.cat(bag_label[i])
        else:
            view_data[i] = None
            bag_label[i] = None
    return view_data[0],view_data[1], bag_size[0], bag_size[1], bag_label[0]

def test_is_valid(data, view_num):
    instance1_len = torch.FloatTensor(data[0].astype(float)).shape[0]
    instance2_len = torch.FloatTensor(data[1].astype(float)).shape[0]
    try:
        for j in range(view_num):
            instance = torch.FloatTensor(data[j].astype(float))
            if j == 1:
                instance = instance.view(-1,300)
            else:
                instance = instance.view(-1,3,224,224)
            if instance1_len > 20 or instance2_len > 50:
                return False
        return True
    except:
        return False

