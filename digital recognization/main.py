# -*- coding: utf-8 -*-
import random

import pandas as pd
import numpy as np

from data_process import load_data
from torch.utils.data import TensorDataset,DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Res_Block

device=torch.device('cuda')
batch_size=128
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

def dealing_data(x):
    return x/255.0

def evaluate(model,dataloader,criterion):
    total_loss = 0.0
    num_batches = 0.0
    model.eval()
    with torch.no_grad():
        for (x,y) in dataloader:
        #for (x,y) in dataloader:
            #x,adj,y = x.to(device),adj.to(device),y.to(device)
            x,y=x.to(device),y.to(device)
            outputs = model(x)
            #loss = criterion(outputs, y)
            loss = criterion(outputs,y)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches

def train(model,optimizer,train_data,val_data,epochs,criterion):

    best_val_loss = 1e9
    model=model.to(device)
    model.train()
    for e in range(epochs):
        total_train_loss=0.
        total_val_loss=0.
        num_train_batches = 0
        for i,(x,y) in enumerate(train_data):
            x,y=x.to(device),y.to(device)
            outputs=model(x).to(device)
            loss=criterion(outputs,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss+=loss.item()
            num_train_batches+=1

        average_train_loss = total_train_loss / num_train_batches
        average_val_loss = evaluate(model,val_data,criterion)

        print(f"epoch:{e},train_loss:{average_train_loss},val_loss:{average_val_loss}")

        if average_val_loss<best_val_loss:
            best_val_loss=average_val_loss
            torch.save(model.state_dict(), 'save_models/model_best.pt')

    return

if __name__=="__main__":
    model=Res_Block(1,10).to(device)
    optimizer=torch.optim.Adam(params=model.parameters(),lr=5e-4)
    criterion=nn.CrossEntropyLoss()

    #datas,targets=load_data('data/train.csv')
    #datas=dealing_data(datas).astype(np.float32)
    #datas=torch.from_numpy(datas)
    #targets=torch.from_numpy(targets)

    #train_x=datas[0:29400,:,:]
    #train_y=targets[0:29400]
    #val_x=datas[29400:,:,:]
    #val_y=targets[29400:]

    #train_loader=DataLoader(dataset=TensorDataset(train_x,train_y),batch_size=128,shuffle=True)
    #val_loader=DataLoader(dataset=TensorDataset(val_x,val_y),batch_size=128,shuffle=False)

    #train(model,optimizer,train_loader,val_loader,50,criterion)
    #torch.save(model.state_dict(), 'save_models/model_last.pt')

    model2 = Res_Block(1,10)
    model2.load_state_dict(torch.load('save_models/model_best.pt'))
    test_datas,_=load_data('data/test.csv')
    test_datas=dealing_data(test_datas).astype(np.float32)
    test_x = torch.from_numpy(test_datas)
    model2.eval()
    with torch.no_grad():
        test_y = torch.max(model2(test_x), dim=1)[1].numpy()
    submission = pd.read_csv('data/sample_submission.csv')
    submission['Label'] = test_y
    submission.to_csv('submission.csv', index=False)
