# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 20:49:06 2021

@author: anshul
"""

import numpy as np
from sklearn.model_selection import train_test_split
   
def load_data():
    X_data=np.load('./data/X_data.npy')
    y=np.load('./data/y_data.npy')    
    X_train, X_test, y_train, y_test = train_test_split(X_data, y, test_size=0.33, random_state=42, stratify=y)   
    return X_train,y_train, X_test, y_test

def get_training_data(batch_size=32):
    
    T,L,val_T,val_L=load_data()
    
    nums=int(len(T)/batch_size)
    T=np.array_split(T,nums)
    L=np.array_split(L,nums)
    
    return T,L


def get_validation_data():
    T,L,val_T,val_L=load_data() 
    return val_T,val_L
