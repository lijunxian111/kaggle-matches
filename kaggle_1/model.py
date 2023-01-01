# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):

    def __init__(self,embedding_dim, num_filter,
                 filter_sizes, output_dim, dropout=0.2, pad_idx=0):
        super(TextCNN,self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=1, out_channels=num_filter,
                  kernel_size=fs)
            for fs in filter_sizes])
    # in_channels：输入的channel，文字都是1
    # out_channels：输出的channel维度
    # fs：每次滑动窗口计算用到几个单词,相当于n-gram中的n
    # for fs in filter_sizes用好几个卷积模型最后concate起来看效果。

        self.fc = nn.Linear(len(filter_sizes) * num_filter, output_dim)
        self.dropout = nn.Dropout(dropout)


    def forward(self, embedded):
        embedded = embedded.unsqueeze(1)  # [batch size, 1, sent len, emb dim]
    # 升维是为了和nn.Conv2d的输入维度吻合，把channel列升维。
        #print(embedded.shape)
        conved = [F.relu(conv(embedded)) for conv in self.convs]
    #print(conved[0].shape, conved[1].shape, conved[2].shape)
    # conved = [batch size, num_filter, sent len - filter_sizes+1]
    # 有几个filter_sizes就有几个conved
        #print(conved[0].shape)

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]  # [batch,num_filter]
    #print(pooled[0].shape, pooled[1].shape, pooled[2].shape)
        x_cat = torch.cat(pooled, dim=1)
    #print(x_cat.shape)
        cat = self.dropout(x_cat)
    # cat = [batch size, num_filter * len(filter_sizes)]
    # 把 len(filter_sizes)个卷积模型concate起来传到全连接层。

        return self.fc(cat)

