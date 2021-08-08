# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 16:11:33 2019

@author: lijun
"""

from generator import *
import tensorflow as tf
from attention import *
import numpy as np

def attention_generator(img_x,img_y):

     with tf.variable_scope('ag'):

        xall,xt,xr=pattentionnet(image=img_x,reuse=False,name='attentionnet_x2y')
        yall,yt,yr=pattentionnet(image=img_y,reuse=True,name='attentionnet_x2y')

        clear_x= generator_unet(image=img_x,reuse=False,name='generator_clear')
        clear_y= generator_unet(image=img_y, reuse=True, name='generator_clear')
        
        
        fake_x_=clear_x*xt+xr
            
        return clear_x,clear_y,fake_x_,xt,yt,xr,yr,xall,yall

       

        