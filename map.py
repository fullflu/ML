#coding:utf-8

import numpy as np
import pandas as pd
import datetime

#sample framework

cpltr = []
cpdtr = []
pred_all = []#pd.read_csv("sample.csv")
val = cpdtr[cpdtr.i_date > datetime.date(2012,6,16)]
val = val.sort('user_id')
val_ulist = list(set(cpltr['user_id'].values))
actual = list(val.groupby('user_id')['coupon_id'].apply(lambda x:x.tolist()).values)
#n = pred_all.shape[0]
#pred_all.index = range(0,n)
#actual = cpdtr[cpdtr.coupon_id.values]
co = map(lambda x: x in val_ulist ,pred_all['user_id'].values)
pred_sub = pred_all[co]
pred_sub.sort("user_id")
predicted = list(pred_sub.groupby('user_id')['coupon_id'].apply(lambda x:x.tolist()).values)
"""
pre_dic = {}
for i in xrange(n):
    if pred_all[i]['user_id'].values not in pre_dic:
        pre_dic[pred_all[i]['user_id'].values] = [pred_all[i]['coupon_id'].values]
    else:
        pre_dic[pred_all[i]['user_id'].values].append(pred_all[i]['coupon_id'].values)
"""

def apk(actual, predicted, k=5):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)

def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted 
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

map5 = mapk(actual,predicted)
print(map5)
