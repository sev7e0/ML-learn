# coding:utf-8

# import pandas as pd #数据分析、处理
import numpy as np
# https://zhuanlan.zhihu.com/p/22345658
import sklearn.datasets as load_Iris

# 读取数据
dataset = load_Iris.load_iris('D:/Iris.data')

from sklearn.model_selection import train_test_split

# 分割数据 抽取百分之十作为测试数据集 random_state随机数种子
x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.1, random_state=1)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# 输入测试数据
y_res = knn.predict(x_test)
predict = knn.score(x_test, y_test)
print('Score:{:f}'.format(np.mean(y_test == y_res)))
print('Score:{:f}'.format(predict))
