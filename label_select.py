import os
import numpy as np
import pickle

num = 20

path = '/data/yangy/data_prepare/COCO/'
label = np.load(path+'coco_label.npy')

label_select = []
count = np.sum(label,axis = 0)

for i in range(num):
    index = np.argmax(count)
    label_select.append(index)
    count[index] = 0

np.save(path+'label_select.npy',label_select)