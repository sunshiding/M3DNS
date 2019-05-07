import os
import csv
import torch
import numpy as np
from PIL import Image
import pickle
import random

from loader.common import get_imcomplete_data
from loader.common import deal_single_text
from loader.common import get_batch
from loader.label_select import select

error_id = ['4072']

def load_data(path,hp):
    #path = '/data/yangy/data_prepare/iapr/'
    ratio = hp['ratio']
    text_select = np.load(path + 'text_select.npy')
    #label_select = np.load(path + 'label_select.npy')
    label_select = select(path,hp)
    label = {}
    with open(path + "label.csv", "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            #self.label[row[0]]
            temp = np.zeros(255)
            for i in range(1, len(row)):
                temp[int(row[i])] = 1
            #print(temp[self.label_select])
            #exit(-1)
            if np.sum(temp[label_select]) > 0:
                label[row[0]] = torch.FloatTensor(temp[label_select])

    data_id = []
    with open(path + 'train_id_all.txt', 'r') as f:
        for row in f:
            x = row.strip().split()
            if x[0] in error_id:
                continue
            if x[0] in label.keys():
                data_id.append(x[0])
    with open(path + 'test_id.txt', 'r') as f:
        for row in f:
            x = row.strip().split()
            if x[0] in error_id:
                continue
            if x[0] in label.keys():
                data_id.append(x[0])
    data_num = len(data_id)
    print(data_num)
    test_num = int(data_num * 0.3)
    semi_num = int(data_num * 0.21)
    train_num = data_num - test_num - semi_num

    random.shuffle(data_id)
    train_id = data_id[0:train_num]
    semi_id = data_id[train_num:train_num+semi_num]
    test_id = data_id[train_num+semi_num:]
    
    np.save("{}train_id.npy".format(path),train_id)
    np.save("{}semi_id.npy".format(path),semi_id)
    np.save("{}test_id.npy".format(path),test_id)

    def get_single_data(idx):
        target = label[idx]
        file_path = path + '/picture/' + idx + '.jpg'
        img_blcok,text_block = 2,4
        bag1 = img_blcok**2
        bag2 = text_block
        if os.path.exists(file_path):
            try:
                img = Image.open(file_path).convert('RGB').resize((224 * img_blcok, 224 * img_blcok))
                img = np.array(img).transpose(2,0,1)
                img = np.expand_dims(img, axis=0)
            except:
                print("图片无法打开 ", file_path)
        if img is None:
            print(file_path)
            print("未找到图片: ", idx)
        imgs = []
        for i in range(img_blcok):
            for j in range(img_blcok):
                imgs.append(img[:,:, (i * 224):((i+1) * 224), (j * 224):((j+1) * 224)])
        imgs = np.concatenate(imgs)

        file_path = path + '/text/' + idx + '.npy'
        text = np.load(file_path)
        text = text[text_select]
        texts = deal_single_text(text, cut=text_block)
        #print(target)
        #print(target.shape)
        return [imgs, texts, bag1, bag2, target]

    train_data = []
    for idx in train_id:
        single_data = get_single_data(idx)
        train_data.append(single_data)

    semi_data = []
    for idx in semi_id:
        single_data = get_single_data(idx)
        semi_data.append(single_data)

    test_data = []
    for idx in test_id:
        single_data = get_single_data(idx)
        test_data.append(single_data)

    print("train data: ", len(train_data))
    print("semi data: ", len(semi_data))
    print("test data: ", len(test_data))

    text_label_data, img_label_data, text_semi_data, img_semi_data = None, None, None, None
    if hp['ratio'] == 0:
        return [train_data,text_label_data, img_label_data, semi_data,text_semi_data, img_semi_data], test_data
    else:
        all_label_data, text_label_data, img_label_data = get_imcomplete_data(train_data,ratio)
        all_semi_data, text_semi_data, img_semi_data = get_imcomplete_data(semi_data,ratio)
    return [all_label_data, text_label_data, img_label_data, all_semi_data, text_semi_data, img_semi_data], test_data
