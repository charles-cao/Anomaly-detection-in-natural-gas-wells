#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
from sklearn import metrics
import hdf5storage
import os
import time


def main(config):
    def getKey(df, i):
        key = "02"
        if (df["日产水量"][i] <= 2):
            key = "02"
        if (df["日产水量"][i] > 2 and df["日产水量"][i] <= 5):
            key = "25"
        if (df["日产水量"][i] > 5 and df["日产水量"][i] <= 10):
            key = "510"
        if (df["日产水量"][i] >= 10):
            key = "10"
        return key
    dl = np.loadtxt(
        f'{os.path.join(config["input path"], "csv", config["input file"]+".csv")}', delimiter=',')
    data, label = dl[:, :-2], dl[:, -2]
    y_all = label
    min_vals = np.min(data, axis=0)
    max_vals = np.max(data, axis=0)
    data = (data - min_vals) / (max_vals - min_vals)
    df = pd.DataFrame(data, columns=['时间', '油压', '套压', '进站压力', '日产气量', '日产水量'])

    df["10_day_average"] = 0
    for i in range(df.shape[0]):
        if (i > 10):
            df.loc[i, "10_day_average"] = df["日产水量"].iloc[i-10:i].values.sum()/10

    y_predict = []
    skip_column = 0
    dataRefer = {"02": 3, "25": 2, "510": 1.8, "10": 1.5}
    for i in range(df.shape[0]):
        if (i < 10 or i > df.shape[0]-3):
            y_predict.append(0)
        else:
            if (skip_column == 0 and df["日产水量"][i] < df["10_day_average"][i] and df["日产水量"][i+1] < df["10_day_average"][i] and df["日产水量"][i+2] < df["10_day_average"][i]):
                y_predict.append(1)
                skip_column = 2
            elif (skip_column == 0 and df["日产水量"][i] > df["10_day_average"][i]*dataRefer.get(getKey(df, i)) and df["日产水量"][i+1] < df["10_day_average"][i]*dataRefer.get(getKey(df, i+1)) and df["日产水量"][i+2] < df["10_day_average"][i]*dataRefer.get(getKey(df, i+2))):
                y_predict.append(1)
                skip_column = 2
            elif (skip_column > 0):
                y_predict.append(1)
                skip_column = skip_column-1
            else:
                y_predict.append(0)
    np.savetxt(
        f'{config["output path"]}/{config["name"]}_{config["input file"]}_{metrics.roc_auc_score(y_all, y_predict):.8f}_{time.time():.8f}.score', y_predict)
    # print(file[:-4], auc)
