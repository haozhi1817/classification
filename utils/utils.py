'''
Author: HaoZhi
Date: 2022-08-19 16:34:37
LastEditors: HaoZhi
LastEditTime: 2022-08-19 18:07:32
Description: 
'''
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from data.dataset import class_dict

def get_label_dict(data_folder):
    label_dict = {}
    datas = glob.glob(os.path.join(data_folder, '*/*.*'))
    for data in datas:
        label_dict[data.split(os.sep)[-1]] = class_dict[data.split(os.sep)[-2]]
    return label_dict

def read_csv(csv_path):
    datas = pd.read_csv(csv_path)
    datas = dict(zip(datas['path'],datas['label']))
    label_dict = {}
    for k, v in datas.items():
        label_dict[k] = class_dict[v]
    return label_dict

def metric(pred_dict, label_dict):
    sorted_pred = np.array(sorted(pred_dict.items(), key = lambda x: x[0]))
    sorted_label = np.array(sorted(label_dict.items(), key = lambda x: x[0]))
    #sorted_label = list(map(lambda x: class_dict[x], sorted_label[:,1]))
    cm = confusion_matrix(y_true= sorted_label[:,1], y_pred = sorted_pred[:,1])
    return cm


def save_result(pred_dict, label_dict, csv_path):
    result = []
    for k in pred_dict.keys():
        result.append([k, label_dict[k], pred_dict[k]])
    pd.DataFrame(result).to_csv(csv_path)



