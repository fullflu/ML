#coding:utf-8

import numpy as np
import pandas as pd
import datetime
#not completed

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


