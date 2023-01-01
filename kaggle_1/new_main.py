import numpy as np
import pandas as pd
import re
#读取数据集
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
#写清洗函数
def clean_text(rawText):
    rawText[:] = [re.sub(r'https?:\/\/.*\/\w*', 'URL', text) for text in rawText]
    rawText[:] = [re.sub(r'@\w+([-.]\w+)*', '', text) for text in rawText]
    rawText[:] = [re.sub(r'&\w+([-.]\w+)*', '', text) for text in rawText]
#调用函数就地工作
clean_text(train_df['text'])
clean_text(test_df['text'])

from sklearn import feature_extraction,model_selection,preprocessing
count_vectorizer = feature_extraction.text.CountVectorizer()      #创建转换器
example_train_vectors = count_vectorizer.fit_transform(train_df["text"][0:5]) #构建示例转换向量
#print(example_train_vectors[0].todense().shape)  #打印出示例的维度
#print(example_train_vectors[0].todense())            #打印出示例向量具体内容
train_vectors = count_vectorizer.fit_transform(train_df["text"])  #进行训练集文本的拟合和转换
test_vectors = count_vectorizer.transform(test_df["text"])       #进行测试集文本的转换

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB
y_label = train_df['target']   #提取标签
X_train, X_test, y_train, y_test = train_test_split(train_vectors, y_label, test_size=0.1, random_state=7) #划分训练集和测试集
clf = BernoulliNB() #构建朴素贝叶斯分类器
clf.fit(train_vectors,y_label)  #拟合模型

sample_submission = pd.read_csv("sample_submission.csv")
Y = clf.predict(test_vectors)
sample_submission["target"] = Y
sample_submission.head()
sample_submission.to_csv("submission17.csv", index=False)
