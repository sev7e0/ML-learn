import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as mlt
import time
# t1=time.time()
train = pd.read_csv('D:/BaiduNetdiskDownload/new_data/1.csv')
# test = pd.read_csv('D:/BaiduNetdiskDownload/new_data/test_set.csv')
# test_id = pd.read_csv('D:/BaiduNetdiskDownload/new_data/test_set.csv')[["id"]].copy()
# train.cumsum()
# train.plot()
ax = train.plot.scatter(x='id',y='class',color='DarkBlue',label='Class1')
train.plot.scatter(x='A',y='C',color='LightGreen',label='Class2',ax=ax)
mlt.show()
print(train)





# column="word_seg"
# n = train.shape[0]
# vec = TfidfVectorizer(ngram_range=(1,2),min_df=3, max_df=0.9,use_idf=1,smooth_idf=1, sublinear_tf=1)
# trn_term_doc = vec.fit_transform(train[column])
# test_term_doc = vec.transform(test[column])
#
# y=(train["class"]-1).astype(int)
# clf = LogisticRegression(C=4, dual=True)
# clf.fit(trn_term_doc, y)
# preds=clf.predict_proba(test_term_doc)
#
# #保存概率文件
# test_prob=pd.DataFrame(preds)
# test_prob.columns=["class_prob_%s"%i for i in range(1,preds.shape[1]+1)]
# test_prob["id"]=list(test_id["id"])
# test_prob.to_csv('D:/BaiduNetdiskDownload/new_data/prob_lr_baseline.csv',index=None)
#
# #生成提交结果
# preds=np.argmax(preds,axis=1)
# test_pred=pd.DataFrame(preds)
# test_pred.columns=["class"]
# test_pred["class"]=(test_pred["class"]+1).astype(int)
# print(test_pred.shape)
# print(test_id.shape)
# test_pred["id"]=list(test_id["id"])
# test_pred[["id","class"]].to_csv('D:/BaiduNetdiskDownload/new_data/sub_lr_baseline.csv',index=None)
# t2=time.time()
# print("time use:",t2-t1)