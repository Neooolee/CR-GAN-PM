# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:08:04 2018

@author: Neoooli
"""

from __future__ import print_function
 
import argparse
from datetime import datetime
from random import shuffle
import random
import os
import sys
import time
import math
import tensorflow as tf
import numpy as np
import glob
from PIL import Image
from attention import *
from generator import *
from batch_read_img import *
from gdaldiy import *
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--x_test_data_path", default='E:/lijun/data/thincloud/test/rgbnclips/cloud/', help="path of x test datas.") #x域的测试图片路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--bands", type=int, default=4, help="load batch size") #batch_size
parser.add_argument("--batch_size", type=int, default=1, help="load batch size")
parser.add_argument("--snapshots", default='./snapshots/',help="Path of Snapshots") #读取训练好的模型参数的路径
parser.add_argument("--out_dir", default='./test_out/image/',help="Output Folder") #保存x域的输入图片与生成的y域图片的路径
args = parser.parse_args()

def make_test_data_list(data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath= glob.glob(os.path.join(data_path, "*")) #读取全部的x域图像路径名称列表
    image_path_lists=[]
    for i in range(len(filepath)):
         path=glob.glob(os.path.join(filepath[i], "*"))
         for j in range(len(path)):
             image_path_lists.append(path[j]) 
    return image_path_lists

def main(num):
    tf.reset_default_graph()

    if not os.path.exists(args.out_dir): #如果保存x域测试结果的文件夹不存在则创建
        os.makedirs(args.out_dir)      
    x_datalists= make_test_data_list(args.x_test_data_path) #得到待测试的x域和y域图像路径名称列表
    test_x_image = tf.placeholder(tf.float32,shape=[args.batch_size, args.image_size, args.image_size,args.bands], name = 'test_x_image') #输入的x域图像
    with tf.variable_scope('ag'):
        dehazed=generator_unet(image=test_x_image,reuse=False,name='generator_clear')
   
    restore_var = [v for v in tf.global_variables() if 'ag/generator_clear' in v.name] #需要载入的已训练的模型参数
 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #建立会话层
    
    saver = tf.train.Saver(var_list=restore_var) #导入模型参数时使用
    ckpt = tf.train.get_checkpoint_state(args.snapshots)
    modelname=args.snapshots+'model-'+num
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, modelname) #导入模型参数
    scale=255
    for i in range(len(x_datalists)):
        starttime=datetime.now()     
        out_path=x_datalists[i].split('\\')
        testx = read_img(x_datalists[i:1+1],scale)
        
        feed_dict = {test_x_image:testx} #建立feed_dict
        
        pre=sess.run(dehazed,feed_dict=feed_dict) #得到生成的y域图像与x域图像         
        write_image=pre[0,:,:,:]*scale
        if not os.path.exists(args.out_dir+out_path[-2]): #如果保存x域测试结果的文件夹不存在则创建
            os.makedirs(args.out_dir+out_path[-2])
        
        savepath=args.out_dir+out_path[-2]+'/'+out_path[-1].split('.')[-2]+'.tif'
        imgwrite(savepath,write_image)
        print('Done')
        

if __name__ == '__main__': 
    main('400000')

