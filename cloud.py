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
from evaluate import *
parser = argparse.ArgumentParser(description='')
 
parser.add_argument("--x_test_data_path", default='E:/lijun/data/thincloud/test/cloudrgbnclipss/', help="path of x test datas.") #x域的测试图片路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--bands", type=int, default=4, help="load batch size") #batch_size
parser.add_argument("--batch_size", type=int, default=1, help="load batch size")
parser.add_argument("--snapshots", default='./snapshots/',help="Path of Snapshots") #读取训练好的模型参数的路径
parser.add_argument("--out_dir", default='./test_out/cloud/',help="Output Folder") #保存x域的输入图片与生成的y域图片的路径
args = parser.parse_args()

def make_test_data_list(data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath= glob.glob(os.path.join(data_path, "*")) #读取全部的x域图像路径名称列表
    image_path_lists=[]
    for i in range(len(filepath)):
         path=glob.glob(os.path.join(filepath[i], "*"))
         for j in range(len(path)):
             image_path_lists.append(path[j]) 
    return image_path_lists
def acv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = img*255.0
    return img_rgb.astype(np.float32) #
def get_write_picture(x_image,t,fake_y,a,r,fake_x_): #get_write_picture函数得到训练过程中的可视化结果
    x_image = acv_inv_proc(x_image[0]) #还原x域的图像
    t = acv_inv_proc(t[0]) #还原y域的图像
    fake_y = acv_inv_proc(fake_y[0])
    a = acv_inv_proc(a[0])
    r = acv_inv_proc(r[0]) #还原x域的图像
    fake_x_ = acv_inv_proc(fake_x_[0]) #还原y域的图像
   #还原生成的y域的图像
    row1 = np.concatenate((x_image,t,fake_y), axis=1) #得到训练中可视化结果的第一行
    row2 = np.concatenate((a,r,fake_x_), axis=1) #得到训练中可视化结果的第二行
    output = np.concatenate((row1, row2), axis=0) #得到训练中可视化结果
    return output
def main(num):
    tf.reset_default_graph()

    if not os.path.exists(args.out_dir): #如果保存x域测试结果的文件夹不存在则创建
        os.makedirs(args.out_dir)      
    x_datalists= make_test_data_list(args.x_test_data_path) #得到待测试的x域和y域图像路径名称列表
    test_x_image = tf.placeholder(tf.float32,shape=[args.batch_size, args.image_size, args.image_size,args.bands], name = 'test_x_image') #输入的x域图像
    with tf.variable_scope('ag'):
        xall,xt,xr=pattentionnet(image=test_x_image,reuse=False,name='attentionnet_x2y')
        clear_x= generator_unet(image=test_x_image,reuse=False,name='generator_clear')
        fake_x_=clear_x*xt+xr
        xa=tf.ones_like(xr)-xr-xt
    restore_var = [v for v in tf.global_variables() if 'ag' in v.name] #需要载入的已训练的模型参数
 
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
        print('开始时间：',starttime)
        out_path=x_datalists[i].split('\\')
        testx = read_img([x_datalists[i]],scale)
        
        feed_dict = {test_x_image:testx} #建立feed_dict
        
        fake_y_value,fake_x_value_,tx,rx,ax= sess.run([clear_x[:,:,:,0:3],fake_x_[:,:,:,0:3],xt[:,:,:,0:3],xr[:,:,:,0:3],xa[:,:,:,0:3]], feed_dict=feed_dict) #run出网络输出
        
        write_image = get_write_picture(testx[:,:,:,0:3],tx,fake_y_value,ax,rx,fake_x_value_) #得到训练的可视化结果
        if not os.path.exists(args.out_dir+out_path[-2]): #如果保存x域测试结果的文件夹不存在则创建
            os.makedirs(args.out_dir+out_path[-2])
        write_image_name = args.out_dir+out_path[-2]+'/'+out_path[-1].split('.')[-2]+'.tif' #待保存的训练可视化结果路径与名称
        Image.fromarray(np.uint8(write_image)).save(write_image_name)

if __name__ == '__main__': 
    main('400000')