#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 13:28:17 2020

@author: elijahchandler
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn import datasets
import tensorflow as tf

df = tf.keras.datasets.mnist.load_data()

#%%
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
X_train = np.reshape(X_train, (X_train.shape[0], 28*28))
X_test = np.reshape(X_test, (X_test.shape[0], 28*28))

#Augment the feature vectors:, 28*28
#%%
X_train = X_train/255.0
X_test = X_test/255.0

N,d = X_train.shape
allOnes = np.ones((N,1))
X_train = np.hstack((allOnes, X_train))

#%%softmax function
def softmax(u):
    expu = np.exp(u)
    return expu/np.sum(expu)

#%%    
# do i need to one hot encode here
#In log reg am i only looking for beta
# why is this so stressful

maxIter = 20
#learning rate/step size\
#slower progress because step size is smaller
#line search tool can adaptively change alpha for neural networks
alpha = .00001



def logReg(X,Y,alpha):
#gradient descent
    N = X.shape[0]
    k = Y.shape[1]
    beta = np.zeros((k , d+1))
    gradNorms = []
    
    for idx in range(maxIter):
        
        grad = np.zeros((k,d+1))
    
        for i in range(N):
            XiHat = X_train[i,:]
            Yi = Y[i,:]
            
        #dot product: @ means matrix multiplication in numpy
        
            u = beta @ XiHat
            
            Su = softmax(u)
            
            grad_i = np.zeros((k, d+1))
            
            for a in range(k):
                grad_i[a,:] = (Su[a] - Yi[a]) * XiHat
            
            grad += grad_i
        
        
    #recompute beta
        beta = beta-alpha*grad
    
    #find the norm 
        nrm = np.linalg.norm(grad)
        gradNorms.append(nrm)
        
        print(idx, nrm)
    return beta, gradNorms 
    
#gradient norms should get closer and closer to 0 

#%%

N_test = X_test.shape[0]
allOnes = np.ones((N_test, 1))

X_test = np.hstack((allOnes,X_test))
#%%

def label():
    correct = 0
    for i in range(N_test):
        XiHat = X_test[i, :]
        Yi = Y_test[i]
        u = beta @ XiHat
        Su = softmax(u)
        k = np.argmax(Su)                    
        if k ==0:
            pred = '0 '
        if k ==1:
            pred = "1"
        if k ==2: 
            pred = "3"
        if k ==3:
            pred = '0'
        if k ==4:
            pred = "1"
        if k ==5: 
            pred = "3"            
        if k ==6:
            pred = '0'
        if k ==7:
            pred = "1"
        if k ==8: 
            pred = "3"
        if k ==9:
            pred = '0'

        #print('pred: ' + pred)
        if(k == Yi):
            correct+=1
           
    print(correct/N_test)
            
 #%%
Y_oneHot = pd.get_dummies(Y_train).values
beta, gradNorms = logReg(X_train, Y_oneHot,alpha)
plt.semilogy(gradNorms)
label()