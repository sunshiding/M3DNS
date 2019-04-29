import os
import torch
import numpy as np
from PIL import Image
import pickle
import csv

def get_imcomplete_data(data,ratio):
    ratio = ratio / 2
    num = len(data)
    imcomplete_num = int(num * ratio)
    data_id = list(range(len(num)))
    random.shuffle(data_id)
    img_id = data_id[:imcomplete_num]
    text_id = data_id[imcomplete_num:imcomplete_num*2]
    all_id = data_id[imcomplete_num*2:]

    all_data = [data[i] for i in all_id]
    img_data = [data[i] for i in img_id]
    text_data = [data[i] for i in text_id]
    return all_data, img_data, text_data

def deal_single_text(x, cut=3):
    d = x.shape[0]
    s = 0
    bag = []
    for i in range(cut):
        dim = int(d / (cut - i))
        text = np.zeros((x.shape[0]))
        text[s:(s + dim)] = x[s:(s+dim)]
        bag.append(text.reshape(1, -1))
        d -= dim
        s += dim
    bag = np.concatenate(bag)
    #bag = torch.FloatTensor(bag)
    return bag

def get_batch(data,indices):
    x_img = []
    x_text = []
    bags1 = []
    bags2 = []
    ys = []
    for idx in indices:
        img,text,bag1,bag2,y = data[idx]
        x_img.append(img)
        x_text.append(text)
        bags1.append(bag1)
        bags2.append(bag2)
        ys.append(y.reshape((1,-1)))
    x_img = np.concatenate(x_img)
    x_img = torch.FloatTensor(x_img)
    x_text = np.concatenate(x_text)
    x_text = torch.FloatTensor(x_text)
    ys = np.concatenate(ys)
    ys = torch.FloatTensor(ys)
    return x_img,x_text,bags1,bags2,ys
