#coding:utf-8
'''
Copyright (C) @fullflu

'''
import numpy as np
import scipy as sp
from scipy import  linalg
import time
import matplotlib.pyplot as plt
from math import *
from sklearn.cross_validation import KFold,ShuffleSplit, cross_val_score
from sklearn.utils.validation import check_arrays, check_random_state
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix

import random



"""
--terminal command

import minibatchSGD
print("mean loss is{}".format(minibatchSGD.run()))

#you can input some argument of run(), but when you input data, one of d or filename has to be kept None
#and you can chaneg the type of Passive and Aggressive method

"""



#-----prepare template text, you dont need to use this

def write_tempfile(filename):
    """
    write a template file to check  the method "minibatch_input"
    """
    #filename="temp.txt"
    f=open(filename,"wt")
    #write the matrix [[0,0,1],[1,1,1],[2,2,0],[3,3,1],[4,4,1],[5,5,1],[6,6,0]] to a text file
    lo="0\t0\t1\n1\t1\t1\n2\t2\t0\n3\t3\t1\n4\t4\t1\n5\t5\t1\n6\t6\t0"
    f.write(lo)
    f.close()

#--------- 

   
#---- get sample    

def get_sample(N=30,dimension=3,type="pa_1"):
    data = np.empty((N,dimension+1))
    data[:N/2,:dimension-2] = random.uniform(-1,1) +5
    data[N/2:,:dimension-2] = random.uniform(-1,1) -5
    data[:,dimension-2] = random.uniform(-1,1)*5
    data[:,dimension-1] = 1
    data[:N/2,-1] = 1
    data[N/2:,-1] = -1
    return data

def get_number(d=None,filename=None,type="pa_1"):
    
    if filename!=None:
        with open(filename) as f:
            dimension=len(f.readline)-1
            N = sum(1 for l in f)
        return [N,dimension]
    
    elif d!=None:
        dimension=sum(1 for l in d[0])
        N = sum(1 for l in d)
        return [N,dimension]
    
    else:
        #get test sample size
        return [30,3]

#------


#----input functions
def minidata_input(dt):
    dt=np.array(dt)
    xt=dt[:,:-1]
    yt=dt[:,-1]
    
    return [xt,yt]

def minibatch_input(filename,random_indices,minibatch_size,dimension,t):
    """
    input minibatch data online from a text file "filename"
    """
    i=0
    j=0
    xt=[]
    yt=[]
    with open(filename) as f:
        for a in f:
            if i in random_indices[t:t+minibatch_size]:
                d=a.strip().split("\t")
                d=[float(q) for q in d]
                xt.append(d[:dimension])
                yt.append(d[-1])
                j+=1
                
            if j > minibatch_size:
                break
            
            i+=1
            
    return [xt,yt]    

#-----



#----training function

def pa_1(indices,minibatch_size,d,f,dimension,lam):
    n=len(indices)
    w = np.empty(dimension)#coefficient of linear model
    max_iter=n
    
    for t in xrange(max_iter):
        
        #randomize the index to get minibatch data
        random_indices = random.sample(indices,n)
        
        #input minibatch or minidata
        if d!=None:
            dt = d[random_indices[t:t + minibatch_size]]
            [xt,yt] = minidata_input(dt)
        
        else:
            [xt,yt] = minibatch_input(filename,random_indices,minibatch_size,dimension,t)
        
        g = np.empty(dimension)
        for k in xrange(minibatch_size):
            #pa_class=PA(xt,yt,w,t,lam)
            #g += pa_class.fit(type)
            g += 1.0*yt[k]*max(0,1-xt[k].dot(w)*yt[k])/(xt[k].dot(xt[k])+lam)*xt[k]
            g /= minibatch_size
        
        """
        #judge convergence
        if w.dot(w+g)<tol:
            w += g
            break 
        """
          
        w += g
        
        
    return w


def pa_2():
        
    return True
    
def pa_regression():
        
    return True

#----


#----loss evaluation

def pa_1_loss(test,w,d,filename):
    
    n = len(test)
    minibatch_size=1
    loss=0
    
    for t in xrange(n):
        random_indices = random.sample(test,n)
        if d!=None:
            dt = d[random_indices[t:t + minibatch_size]]
            [xt,yt] = minidata_input(dt)
        
        else:
            [xt,yt] = minibatch_input(filename,random_indices,minibatch_size,dimension,t)
        
    loss += max(0,1 - yt*xt.dot(w))
    loss /= n
    
    return loss
    
def pa_2_loss(test,w,d,filename):
    
    pass

def pa_regression(test,w,d,filename):
    
    pass

#------------        



#-----Class of Passive and Aggressive algorithm

class PA(object):
    def __init__(self,type,lam):
        #self.xt=xt
        #self.yt=yt
        #self.w=w
        #self.t=t
        self.lam=lam
        self.type=type
         
    def fit(self,indices,minibatch_size,d,filename,dimension):
        
        if self.type=="pa_1":
            return pa_1(indices,minibatch_size,d,filename,dimension,self.lam)
            #return 1.0*self.yt*max(0,1-self.w.dot(self.xt)*self.yt)/(self.xt.dot(self.xt)+self.lam)*self.xt
        
        elif self.type=="pa_2":
            return True
        
        else:
            return True
        
    def score(self,test,w,d,filename):
        
        if self.type=="pa_1":
            return pa_1_loss(test,w,d,filename)
        
        elif self.type =="pa_2":
            return True
        
        else:
            return True


#-----    



#----- run 
def run(d=None,filename=None,type="pa_1",n_folds=5,minibatch_size=1,lam=0.1):
    
    [N,dimension] = get_number(d, filename,type)
    cv = KFold(N,n_folds=n_folds,shuffle=True)
    if d==None and filename==None:
        d = get_sample(N=30,dimension=3,type=type)
        
    mean_loss = 0
    for indices,test in cv:
        pa_class=PA(type,lam)
        w = pa_class.fit(indices,minibatch_size,d,filename,dimension)
        mean_loss += pa_class.score(test, w, d, filename)
        
    mean_loss /= n_folds
    
    return mean_loss
        
#------

