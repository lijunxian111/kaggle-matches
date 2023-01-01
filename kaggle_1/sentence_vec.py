# -*- coding: utf-8 -*-
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pandas as pd
import numpy as np
from data_process import process_data


# 构建Doc2vec模型，获得句子向量
def get_sentence_vec(datasets):
    # gemsim里Doc2vec模型需要的输入为固定格式，输入样本为[句子，句子序号]
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(datasets)]
    # 初始化和训练模型
    model = Doc2Vec(documents, vector_size=50, dm=1, window=4, min_count=5, epochs=50)
    # model = Doc2Vec(vector_size=300, dm=1, window=4, min_count=5, epochs=50)
    # model.build_vocab(documents)
    # model.train(documents,total_examples=model.corpus_count,epochs=model.epochs)

    model.save('data/w2v/doc2vec_model.pkl')  # 将模型保存到磁盘
    # 获得数据集的句向量
    documents_vecs = np.concatenate([np.array(model.docvecs[sen.tags[0]].reshape(1, 50)) for sen in documents])
    #np.save('data/w2v/feature_vectors.npy',documents_vecs)
    np.save('data/w2v/feature_vectors.npy', documents_vecs)
    return documents_vecs

def load_data(path):
    with open(path,'r+') as f:
          lines=pd.read_csv(f)

    datas=lines.values
    data=datas[:,1:4]
    for item in data:
        item[2]=process_data(item[2])

    if len(datas[0])==5:
        target=lines['target']
        #print(target)
        return data[:,2],target.to_numpy()
    else:
        return data[:,2],None


if __name__ == '__main__':
    # 准备数据
    train_data,_=load_data('train.csv')
    test_data,_=load_data('test.csv')
    #print(len(train_data))
    #print(len(test_data))

    datasets=list(train_data)+list(test_data)

    #cw = lambda x: str(x).split()
    #train_data['words'] = train_data['contents'].apply(cw)
    #test_data['words'] = train_data['contents'].apply(cw)
    #datasets = pd.concat([train_data, test_data])

    # doc2vec句向量训练和生成
    documents_vec = get_sentence_vec(datasets)
    #print(documents_vec[0])

    # 加载训练好的模型
    doc2vec_model = Doc2Vec.load('data/w2v/doc2vec_model.pkl')
    #documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(datasets)]
    #print(doc2vec_model.voc)
    # 推断新文档向量
    #doc2vec_model.infer_vector(['绝望', '快递', '说', '收到', '快递', '中奖', '开心'])