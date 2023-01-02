# -*- coding: utf-8 -*-
import torch
import torch.nn as nn

class Res_Block(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(Res_Block,self).__init__()

        self.in_channels=1
        self.conv1=nn.Conv2d(in_channels=self.in_channels,out_channels=16,kernel_size=3,stride=1,padding=1)
        self.bn1=nn.BatchNorm2d(16)
        self.pool1=nn.MaxPool2d(2)
        self.conv2=nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu=nn.ReLU()
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3=nn.BatchNorm2d(64)
        self.downsample=nn.Conv2d(1,64,kernel_size=3,stride=1,padding=1)
        self.pool2=nn.MaxPool2d(2)
        self.avgpool=nn.AvgPool2d(2)
        self.fc1=nn.Linear(64*7*7,512)
        self.fc2=nn.Linear(512,num_classes)

    def forward(self,x):
        in_size=x.shape[0]
        x=x.unsqueeze(1)
        h=x
        x=self.bn1(self.relu(self.conv1(x)))
        x=self.pool1(x)
        x=self.bn2(self.relu(self.conv2(x)))
        x=self.bn3(self.relu(self.conv3(x)))
        identity=self.downsample(h)
        identity=self.pool2(self.bn3(identity))
        #print(x.shape)
        #print(identity.shape)
        x=x+identity
        x=self.avgpool(x).view(in_size,-1)
        #print(x.shape)
        x=self.relu(self.fc1(x))
        x=self.fc2(x)
        return x


