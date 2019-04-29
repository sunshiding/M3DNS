import os
import numpy as np
import pickle
import csv

def select(path,hp):
    num = hp['label']
    if hp['dataname'].lower() == "coco":
        label = np.load(path+'coco_label.npy')
        count = np.sum(label,axis = 0)
    elif hp['dataname'].lower() == "flickr":
        label = np.load(path + 'label_data.npy')
        count = np.sum(label,axis = 0)
    elif hp['dataname'].lower() == "nus":
        label = np.load(path+'label.npy')
        count = np.sum(label,axis = 0)
    elif hp['dataname'].lower() == "iapr":
        error_id = ['4072']
        count = np.zeros(255)
        with open(path + "label.csv", "r") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                for i in range(1, len(row)):
                    count[int(row[i])] += 1

    label_select = []
    for i in range(num):
        index = np.argmax(count)
        label_select.append(index)
        count[index] = 0
    label_select.sort()
    print(label_select)
    np.save(hp['rootdir']+'label_select.npy',label_select)

    return label_select
