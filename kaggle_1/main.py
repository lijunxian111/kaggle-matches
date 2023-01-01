import numpy as np
import pandas as pd
import random
from gensim.models.doc2vec import Doc2Vec
from data_process import process_data
from sentence_vec import load_data
from model import TextCNN
#from keras.models import load_model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset,TensorDataset

random.seed(42)
device=torch.device('cuda')

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

def test():
    return


if __name__=="__main__":
    _,target=load_data('train.csv')
    train_vectors=np.load('data/w2v/feature_vectors.npy')
    test_vectors=np.load('data/w2v/test_vectors.npy')
    #print(type(target[0]))


    train_y=torch.from_numpy(target[:6400])
    train_dts=torch.from_numpy(train_vectors[:6400,:])
    val_y=torch.from_numpy(target[6400:])
    val_dts = torch.from_numpy(train_vectors[6400:7613, :])

    train_dataset=TensorDataset(train_dts,train_y)
    val_dataset = TensorDataset(val_dts, val_y)

    train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
    val_loader=DataLoader(val_dataset,batch_size=16,shuffle=False)

    model=TextCNN(50,100,[2,3,4],2)
    optimizer=torch.optim.Adam(params=model.parameters(),lr=5e-4)
    criterion=nn.CrossEntropyLoss()
    train(model,optimizer,train_loader,val_loader,50,criterion)
    torch.save(model.state_dict(),'save_models/model_last.pt')


    model2=TextCNN(50,100,[2,3,4],2)
    model2.load_state_dict(torch.load('save_models/model_best.pt'))
    test_x=torch.from_numpy(train_vectors[7613:,:])
    test_y=torch.max(model2(test_x),dim=1)[1].numpy()
    print(test_y)
    submission=pd.read_csv('sample_submission.csv')
    submission['target']=test_y
    submission.to_csv('submission.csv',index=False)

    #print(loader)
    #print(data,target)
    #print(data[2][1])
    #print(process_data(data[2][2]))
    #lst=[]
    #for i in range(len(data)):
        #if data[i][1] not in lst:
            #print(data[i][1])
            #lst.append(data[i][1])

