# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 15:13:17 2018

@author: lijun
"""

import numpy as np
from gdaldiy import *

def randomflip(input_,n):
    #生成-3到2的随机整数，-1顺时针90度，-2顺时针180，-3顺时针270,0垂直翻转，1水平翻转，2不变
    if n<0:
        return np.rot90(input_,n)
    elif -1<n<2:
        return np.flip(input_,n)
    else: 
        return input_
def read_img(datapath,scale=255,k=2):
    img_list=[]
    l=len(datapath)
    for i in range(l):
        img=imgread(datapath[i])/scale
        img = randomflip(img,k)
        img=img[np.newaxis,:]
        img_list.append(img)    
    imgs=np.concatenate(img_list,axis=0)
    return imgs




