#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 00:55:09 2017

@author: hakozaki
"""
import numpy as np
import matplotlib.pyplot as plt

import itertools

class kMeans:
    
    def __init__(self,k):
        u0_x = 0.5 * np.random.rand(1)
        u0_y = 0.5 * np.random.rand(1) + 0.5

        u1_x = 0.5 * np.random.rand(1) + 0.5
        u1_y = 0.5 * np.random.rand(1)
        
        u0 = np.c_[u0_x,u0_y]
        u1 = np.c_[u1_x,u1_y]
        
        self.u = np.r_[u0,u1]
 
    def J(self,X,L):
        sum_value = 0.0
        for x,l in zip(X,L):
            distance = np.sum( (x - self.u)**2 , axis = 1)
            sum_value += np.sum(l * distance)
        return sum_value
        

if __name__ == '__main__':
    
    x = 0.5 * np.random.rand(10)
    y = 0.5 * np.random.rand(10)
    
    x2 = 0.5 * np.random.rand(10) + 0.5
    y2 = 0.5 * np.random.rand(10) + 0.5
    
    data1 = np.c_[x,y]
    data2 = np.c_[x2,y2]

    datas = np.r_[data1,data2]

    N = len(datas)
    K = 2
    
    label = np.zeros([N,K])   
    #初期値はゼロ番目のクラスタに属すると仮定
    label[: , 0] = 1
    
    k_means = kMeans(K)

    ### グラフ描写 before ###
    fig = plt.figure()

    ax = fig.add_subplot(2,1,1)
    ax.scatter(datas[:,0],datas[:,1], c='red')
    ax.scatter(k_means.u[:,0],k_means.u[:,1],marker='s',color='y')
    
    
    target = k_means.J(datas,label)
    
    diff = 1000
    
    for turn in itertools.takewhile(lambda turn: diff > 0.01, itertools.count()):
    
        print('turn : %d , loss : %.3f' % (turn , target))
    
        ### Eステップ ###
        for row , d in enumerate(datas):
            
            #k個のクラスタの中心との距離のリスト
            distance = np.sum( (d - k_means.u)**2 , axis = 1)
                
            idx = np.argmax(distance)
            
            for k in range(len(k_means.u)):    
                label[row , k] = 1.0 if k == idx else 0
    
        
        ### Mステップ ###
        for k in range(K):
            
            numerator = 0.0
            denominator = np.sum(label[:,k])
            
            for row , d in enumerate(datas):
                numerator += d * label[row , k]
            
            k_means.u[k] = numerator / denominator
        
        #収束判定
        new_target = k_means.J(datas,label)
        diff = target - new_target
        target = new_target 
        
    
    ### グラフ描写 after ###
    color = ['g','b','r']
    ax = fig.add_subplot(2,1,2)
    for row , d in enumerate(datas):
        idx = np.argmax(label[row])
        ax.scatter(d[0],d[1],color=color[idx])
    ax.scatter(k_means.u[:,0],k_means.u[:,1],marker='s',color='y')
    
        
        
    
    
    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
    
    
    
    
   
    
    