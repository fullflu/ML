#coding:utf-8

import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
import datetime
#not completed
#can replace U.dot(v) in top5() function

#---
cpdtr = pd.DataFrame()#rewrite here
cpltr = pd.DataFrame()#rewrite here
all_data = pd.DataFrame()#rewrite here
#can add another feature here
len_tr = 3#rewrite here
#---
train_data = all_data[:len_tr]
test_data = all_data[len_tr:]
train_ulist = list(set(train_data['user_id'].values))
train_ulist.sort()

#---#explicit feedback ver
train_data['y'] = 1.0
w = train_data.groupby('user_id')
trainlist = []
y = []
for name,group in w:
    temp = group.merge(cpltr,how="right",on="coupon_id")
    temp['u'] = name
    #del(temp['date'])
    temp = temp.fillna(0)
    y = y + list(temp['y'])
    temp = temp.drop('y',axis=1)
    o = [temp.iloc[l,:].T.to_dict() for l in range(len(temp))]
    trainlist = trainlist + o
    
v = DictVectorizer()
X = v.fit_transform(trainlist)
y = np.array(y)
#y = np.repeat(1.0,X.shape[0])
fm = pylibfm.FM()
fm.fit(X,y)
#fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
fm.predict(v.transform(test_data)) #maybe , we should write --for i in xrange(test_data.shape[0])...-- ?

#inplicit feedback ver -- put at r41f
trainlist = [train_data.iloc[l,:].T.to_dict() for l in range(train_data.shape[0])]
v = DictVectorizer()
X = v.fit_transform(trainlist)
y = np.ones(len_tr)
fm = pylibfm.FM()
fm.fit(X,y)
#fm.predict(v.transform({"user": "1", "item": "10", "age": 24}))
fm.predict(v.transform(test_data)) #maybe , we should write --for i in xrange(test_data.shape[0])...-- ?
"""
new couponとの類似度が高い5クーポンのid,featureを保存
for i in xrange(22765):
    for j in xrange(310):
        td = cplte.iloc[j,:].T.to_dict()
        td['user_id'] = train_ulist[i]
        #fm.predict({'u_id':train_ulist[i], 'c_id':cplte['coupon_id'].values[j], 'feature':;;, ...})
        fm.predict(td)??

"""
"""
cpltr = []
cpdtr = []
pred_all = []
feature_train_list = []
feature_te_list = []
cpdtr = cpdtr.sort('user_id')[feature_train_list]
user_cnt = cpdtr.groupby('user_id')
user_cnt = user_cnt.values[:,0]
train_ulist = list(set(cpdtr['user_id'].values)).sort()
cpdtr.index = cpdtr.shape[0]
cpltr.sort('coupon_id')
#1円とマックカードを抜く前処理
ulist = []
uvec = []
cvec = []
cplte = []
#類似度ベクトルとか特徴量を全部作った後でFM用のDataFraneを作る?
#特徴ベクトルは，userについてのものはuser_idで，couponについてのものはcoupon_idでsortしておく
#なんなら，わざわざUとVをつくらなくてもよいかもしれない
tidx = 0
for i in xrange(len(train_ulist)):
    if i == 0:
        cpfm = pd.merge(cpdtr[tidx: (tidx + user_cnt[i])],cpltr,how='right', on ='coupon_id')
        cpfm = cpfm[feature_train_list]
        #item_countを足したい…
        cpfm.sort(['user_id','coupon_id'])
        cpfm['c_vec'] = cvec
        cpfm['u_vec'] = uvec[i]
        feature_train_list.append('c_vec','u_vec')
        #add test_feature
        ctfm = pd.DataFrame() #長さ指定？
        ctfm['user_id'] = ulist[i]
        ctfm['coupon_id'] = cplte['coupon_id'].values
        #ctfm[tidx: len(cplte['coupon_id'].valies)]['user_id'] = ulist[i] 
        #ctfm[tidx: (tidx + user_cnt[i])]['coupon_id'] = cplte['coupon_id'].values
        
        tidx += user_cnt[i]
    else:
        temp = pd.merge(cpdtr[tidx: (tidx + user_cnt[i])],cpltr,how='right', on ='coupon_id')
        temp = temp[feature_train_list]
        temp.sort(['user_id','coupon_id'])
        temp['u_vec'] = uvec[i]
        temp['c_vec'] = cvec
        cpfm = pd.concat(cpfm,temp)
        tem = pd.DataFrame()
        tem['user_id'] = ulist[i]
        tem['coupon_id'] = cplte['coupon_id'].values
        ctfm = pd.concat(ctfm,tem)
    tidx += user_cnt[i]
    #cpfm.sort(['user_id','coupon_id'])
cpfm.fillna(0)
def bilabel(cpfm):
    return 0
cpfm = bilabel(cpfm)

def fm(cpdm):
    return 0

def answer(model,ctfm):
    return 0

model = fm(cpfm)
answer(model,ctfm)

"""
