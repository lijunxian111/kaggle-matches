# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def load_data(path):
    raw_data=pd.read_csv(path)

    if("label" in raw_data.keys()):
        targets=raw_data['label'].values
        datas=raw_data.values[:,1:]
        return datas.reshape((len(datas),28,28)),targets
    else:
        datas=raw_data.values
        return datas.reshape((len(datas),28,28)),None


if __name__=="__main__":

    datas,targets=load_data('data/train.csv')
    #print(datas.shape)
    #print(datas)
    #print(targets)