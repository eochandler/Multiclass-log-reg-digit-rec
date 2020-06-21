#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 15:55:08 2020

@author: elijahchandler
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd

np.random.seed(7)


def softmax(u):
    expu = np.exp(u)
    return expu/np.sum(expu)

def crossEntropy(p,q):
    return -np.vdot(p,np.log(q))


def eval_L(X, Y, beta):
    N = X.shape[0]
    L = 0.0
    
    for i in range(N):
        XiHat = X[i]
        Yi = Y[i]
        qi = softmax(beta @ XiHat)
        L += crossEntropy(Yi, qi)
        
    return L

def logReg_SGD(X,Y,alpha):
    numEpochs = 5
    N, d = X.shape
    X = np.insert(X,0,1,axis = 1)
    K = Y.shape[1]

    beta = np.zeros((K , d+1))
    Lvals = []
    
    for ep in range(numEpochs):
        
        #these two cost functinos should decres
        L = eval_L(X,Y,beta)
        Lvals.append(L)
        
        print("Epoch is: " + str(ep) + "Cost is: " + str(L))
        
        prm = np.random.permutation(N)
        for i in prm:
            XiHat = X[i]
            Yi = Y[i]
            
            qi = softmax(beta @ XiHat)
            grad_Li = np.outer(qi - Yi, XiHat)
            
            beta = beta-alpha* grad_Li
            
    return beta, Lvals
          
def predictLabels(X, beta):
    X = np.insert(X,0,1, axis =1)
    N = X.shape[0]
    
    predictions = []
    probabilities = []
    
    for i in range(N):
        XiHat = X[i]
        qi = softmax(beta @ XiHat)
        k = np.argmax(qi)
        predictions.append(k)
        probabilities.append(np.max(qi))
        
    return predictions, probabilities

        
    
(X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

X_train = X_train/255.0
X_test = X_test/255.0

N_train, numRows, numCols = X_train.shape
X_train = np.reshape(X_train,(N_train,numRows * numCols))

Y_train = pd.get_dummies(Y_train).values

alpha = .01
       
beta, Lvals =  logReg_SGD(X_train, Y_train,alpha)
        
plt.semilogy(Lvals)       
         
N_test = X_test.shape[0]

X_test = np.reshape(X_test,(N_test,numRows * numCols))

predictions, probabilities = predictLabels(X_test, beta)

probabilities = np.array(probabilities)

agreement = (predictions == Y_test)

sortedIdxs = np.argsort(probabilities)

sortedIdxs = sortedIdxs[::-1]

difficultExamples = []

for i in sortedIdxs:
    
    difficultExamples.append(i)

numCorrect = 0

for i in range(N_test):
    if predictions[i] == Y_test[i]:
        numCorrect+= 1

accuracy = numCorrect/N_test
print('accuracy' + str(accuracy))








