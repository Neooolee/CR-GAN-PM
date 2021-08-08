# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:57:36 2018

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
from batch_read_img import *
from generator import *
from discriminator import *
from attention import *
from attention_generator import *
from allnet import *
parser = argparse.ArgumentParser(description='')
parser.add_argument("--snapshot_dir", default='./snapshots/', help="path of snapshots") #保存模型的路径
parser.add_argument("--out_dir", default='./train_out', help="path of train outputs") #训练时保存可视化输出的路径
parser.add_argument("--image_size", type=int, default=256, help="load image size") #网络输入的尺度
parser.add_argument("--random_seed", type=int, default=1234, help="random seed") #随机数种子
parser.add_argument('--base_lr', type=float, default=0.0002, help='initial learning rate for adam') #基础学习率
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch') #训练的epoch数量
parser.add_argument('--epoch_step', dest='epoch_step', type=int, default=100, help='# of epoch to decay lr') #训练中保持学习率不变的epoch数量
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda") #训练中L1_Loss前的乘数
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument('--beta2', dest='beta2', type=float, default=0.9, help='momentum term of adam') #adam优化器的beta1参数
parser.add_argument("--summary_pred_every", type=int, default=100, help="times to summary.") #训练中每过多少step保存训练日志(记录一下loss值)
parser.add_argument("--write_pred_every", type=int, default=100, help="times to write.") #训练中每过多少step保存可视化结果
parser.add_argument("--save_pred_every", type=int, default=10000, help="times to save.") #训练中每过多少step保存模型(可训练参数)
parser.add_argument("--x_train_data_path", default='E:/lijun/data/thincloud/rgbntrain/rgbncloud/', help="path of cloud  training datas.") #x域的训练图片路径
parser.add_argument("--y_train_data_path", default='E:/lijun/data/thincloud/rgbntrain/rgbnnocloud/', help="path of clear training datas.") #y域的训练图片路径
parser.add_argument("--batch_size", type=int, default=1, help="load batch size") #batch_size
parser.add_argument("--bands", type=int, default=4, help="bands") #batch_size
args = parser.parse_args()
 
def save(saver, sess, logdir, step): #保存模型的save函数
   model_name = 'model' #保存的模型名前缀
   checkpoint_path = os.path.join(logdir, model_name) #模型的保存路径与名称
   if not os.path.exists(logdir): #如果路径不存在即创建
      os.makedirs(logdir)
   saver.save(sess, checkpoint_path, global_step=step) #保存模型
   print('The checkpoint has been created.')

def cv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = (img+1)*127.5
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像
def acv_inv_proc(img): #cv_inv_proc函数将读取图片时归一化的图片还原成原图
    img_rgb = img*255.0
    return img_rgb.astype(np.float32) #返回bgr格式的图像，方便cv2写图像 
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
 
def make_train_data_list(data_path): #make_train_data_list函数得到训练中的x域和y域的图像路径名称列表
    filepath= glob.glob(os.path.join(data_path, "*")) #读取全部的x域图像路径名称列表
    image_path_lists=[]
    for i in range(len(filepath)):
         path=glob.glob(os.path.join(filepath[i], "*"))
         for j in range(len(path)):
             image_path_lists.append(path[j]) #将x域图像数量与y域图像数量对齐
    return image_path_lists
    
def l1_loss(src, dst): #定义l1_loss
    return tf.reduce_mean(tf.abs(src - dst))
def l2_loss(x):
    return tf.sqrt(tf.reduce_sum(x**2))

def gan_loss(src, dst): #定义gan_loss，在这里用了二范数
    return tf.reduce_mean((src-dst)**2)
    
def main():
    tf.reset_default_graph()
    if not os.path.exists(args.snapshot_dir): #如果保存模型参数的文件夹不存在则创建
        os.makedirs(args.snapshot_dir)
    if not os.path.exists(args.out_dir): #如果保存训练中可视化输出的文件夹不存在则创建
        os.makedirs(args.out_dir)
    x_datalists = make_train_data_list(args.x_train_data_path) #得到数量相同的x域和y域图像路径名称列表
    y_datalists = make_train_data_list(args.y_train_data_path)
    tf.set_random_seed(args.random_seed) #初始一下随机数
    x_img = tf.placeholder(tf.float32,shape=[args.batch_size, args.image_size, args.image_size,args.bands],name='x_img') #输入的x域图像
    y_img = tf.placeholder(tf.float32,shape=[args.batch_size, args.image_size, args.image_size,args.bands],name='y_img') #输入的y域图像
    fake_y,clear_y,fake_x_,xt,yt,xr,yr,xall,yall=attention_generator(x_img,y_img)


    dy_real = cnodiscriminator(image=y_img, reuse=False, name='discriminator_y') #判别器返回的对真实的y域图像的判别结果
    dy_fake = cnodiscriminator(image=fake_y, reuse=True, name='discriminator_y') #判别器返回的对生成的y域图像的判别结果
   
    d_loss = (gan_loss(dy_real,tf.ones_like(dy_real)) + gan_loss(dy_fake,tf.zeros_like(dy_fake))) / 2
    
    cross_loss=gan_loss(yt,tf.ones_like(yt))
    exl_loss=(exlusion_loss(fake_y,xt,level=3)+exlusion_loss(fake_y,xr,level=3))
    a_loss = 0.5*cross_loss\
             + args.lamda*l1_loss(x_img,fake_x_)\
             + exl_loss\
             + gan_loss(dy_fake,tf.ones_like(dy_fake))\
             + 0.1*l1_loss(y_img,clear_y)


    xa=tf.ones_like(xr)-xr-xt
    
    tf.summary.scalar("exl_loss", exl_loss)
    tf.summary.scalar("a_g_loss", gan_loss(dy_fake,tf.ones_like(dy_fake)))
    tf.summary.scalar("dis_loss", d_loss) #记录判别器的loss的日志
    
    tf.summary.scalar("a_loss", a_loss) #记录生成器loss的日志
  
    d_vars = [v for v in tf.trainable_variables() if 'discriminator' in v.name] #所有判别器的可训练参数
   
    a_vars = [v for v in tf.trainable_variables() if 'ag' in v.name]
    
    global_step = tf.placeholder(tf.float32, name='step')
    learning_rate = tf.placeholder(tf.float32,name='lr')
    learningrate=decay(global_step,learning_rate)
    tf.summary.scalar('learning_rate', learningrate)

    a_optim = tf.train.AdamOptimizer(learningrate, beta1=args.beta1, name='Adam_AG').minimize(a_loss, var_list=a_vars)
    d_optim = tf.train.AdamOptimizer(learningrate, beta1=args.beta1, name='Adam_D').minimize(d_loss, var_list=d_vars)

    train_op = tf.group(d_optim,a_optim)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(args.snapshot_dir, graph=tf.get_default_graph())
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True #设定显存不超量使用
    sess = tf.Session(config=config) #新建会话层
 
    saver = tf.train.Saver(max_to_keep=1000) #模型保存器
    
    ckpt = tf.train.get_checkpoint_state(args.snapshot_dir)
    if ckpt and ckpt.model_checkpoint_path:
      meta_graph_path = ckpt.model_checkpoint_path + ".meta"
      restore = tf.train.import_meta_graph(meta_graph_path)
      restore.restore(sess, tf.train.latest_checkpoint(args.snapshot_dir))
      filelists= glob.glob(os.path.join(args.out_dir, "*"))
      step = len(filelists)*100
    else:
      sess.run(tf.global_variables_initializer())
      step = 0
    lenx=len(x_datalists)
    leny=len(y_datalists)
    start_epoch=step//leny
    startx=step-(step//lenx)*lenx
    starty=step-(step//leny)*leny
    shuffle(x_datalists)
    scale=255
    for epoch in range(start_epoch,args.epoch): #训练epoch数       
               #每训练一个epoch，就打乱一下x域图像顺序
        shuffle(y_datalists) #每训练一个epoch，就打乱一下y域图像顺序
          
        while (starty+args.batch_size)<=leny:             
            batch_x_image=read_img(x_datalists[startx:startx+args.batch_size],scale,2)
            batch_y_image=read_img(y_datalists[starty:starty+args.batch_size],scale,2) #读取x域图像和y域图像

            feed_dict = {x_img : batch_x_image, y_img : batch_y_image,learning_rate:args.base_lr,global_step:step} #得到feed_dict
            dl,al,_= sess.run([d_loss,a_loss,train_op], feed_dict=feed_dict) #得到每个step中的生成器和判别器loss
            step=step+1
            starty=starty+args.batch_size
            startx=startx+args.batch_size            
            if (startx+args.batch_size)>=lenx:
                shuffle(x_datalists)
                startx=0   

            if step% args.save_pred_every == 0: #每过save_pred_every次保存模型
                save(saver, sess, args.snapshot_dir, step)
            if step% args.summary_pred_every == 0: #每过summary_pred_every次保存训练日志
                summary= sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary, step)

            if step % args.write_pred_every == 0: #每过write_pred_every次写一下训练的可视化结果
                fake_y_value,fake_x_value_,tx,rx,ax,lr= sess.run([fake_y[:,:,:,0:3],fake_x_[:,:,:,0:3],xt[:,:,:,0:3],xr[:,:,:,0:3],xa[:,:,:,0:3],learningrate], feed_dict=feed_dict) #run出网络输出
                write_image = get_write_picture(batch_x_image[:,:,:,0:3],tx,fake_y_value,ax,rx,fake_x_value_) #得到训练的可视化结果
                write_image_name = args.out_dir + "/out"+ str(epoch+1)+'_'+str(step)+ ".png" #待保存的训练可视化结果路径与名称
                Image.fromarray(np.uint8(write_image)).save(write_image_name) #保存训练的可视化结果
                print('epoch step       a_loss       d_loss   lr')
                print('{:d}     {:d}    {:.3f}         {:.3f}   {:.8f} '.format(epoch+1, step,al,dl,lr))
        starty=0
                
if __name__ == '__main__':
    main()

