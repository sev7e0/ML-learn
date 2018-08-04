# coding:utf-8
import numpy as np
# https://zhuanlan.zhihu.com/p/22345658
import sklearn.datasets as load_Iris

dataset = load_Iris.load_iris('D:/Iris.data')

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(dataset[0], dataset[1], test_size=0.1, random_state=1)

import sklearn.svm as svm

sv = svm.SVC(C=0.8, kernel='rbf', gamma=10, decision_function_shape='ovr')

sv.fit(x_train, y_train)

# 输入测试数据
print('Score:{:f}'.format(sv.score(x_test, y_test)))
