#coding:utf-8

import numpy as np
import pandas as pd
import datetime

cpltr = []
cpdtr = []
pw = []
pbd = pd.DataFrame()
last = datetime.date(2012,6,24)
first = datetime.date(2011,6,26)
for j in xrange(52):
    start = last - datetime.timedelta(days=7)
    temp = cpdtr[cpdtr['i_date'] < last]
    #pw.append(temp[temp['i_date'] >= start])
    pw = temp[temp['i_date'] >= start]
    pw['action_id'] = j
    if j == 0:
        pbd = pw
    else:
        pbd = pd.concat(pbd,pw)
    last = start
    
pbd.groupby(['user_id'],['action_id'])['genre_name'].apply(lambda x:x.tolist()).values



#csv? txt?
pbd.to_csv('asseq.csv',index=False)
f = open('asseq.csv','r')
out = f.readlines()
f.close()
f = open('asseq.txt','r')
f.write(out)
