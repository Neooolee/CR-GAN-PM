# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 16:07:54 2018

@author: Neoooli
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
import math
import tensorflow.contrib.slim as slim
#构造可训练参数
def make_var(name, shape, trainable = True):
    return tf.get_variable(name, shape, trainable = trainable)
 
#定义卷积层
def conv2d(input_, output_dim, kernel_size=3, stride=2, padding = "SAME", name = "conv2d", biased = False):
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape=[kernel_size, kernel_size, input_dim, output_dim])
        output = tf.nn.conv2d(input_, kernel, [1, stride, stride, 1], padding = padding)
        if biased:
            biases = make_var(name = 'biases', shape = [output_dim])
            output = tf.nn.bias_add(output, biases)
        return output     
#定义反卷积层
def deconv2d(input_, output_dim, kernel_size=4, stride=2, padding = "SAME", name = "deconv2d"):
    input_dim = input_.get_shape()[-1]
    batchsize=int(input_.get_shape()[0])
    input_height = int(input_.get_shape()[1])
    input_width = int(input_.get_shape()[2])
    with tf.variable_scope(name):
        kernel = make_var(name = 'weights', shape = [kernel_size, kernel_size, output_dim, input_dim])
        output = tf.nn.conv2d_transpose(input_, kernel, [batchsize, input_height * stride, input_width * stride, output_dim], [1, 2, 2, 1], padding = "SAME")
        return output
 
def batch_norms(input_, name="batch_norm"):
    return tf.contrib.layers.batch_norm(input_, decay=0.9, updates_collections=None, epsilon=1e-5,center=True, scale=True, scope=name)

def instance_norm(input_, name="instance_norm"):
    return tf.contrib.layers.instance_norm(input_,scope=name)

#定义最大池化层
def max_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.max_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
#定义平均池化层
def avg_pooling(input_, kernel_size, stride, name, padding = "SAME"):
    return tf.nn.avg_pool(input_, ksize=[1, kernel_size, kernel_size, 1], strides=[1, stride, stride, 1], padding=padding, name=name)
 
#定义lrelu激活层
def lrelu(x, leak=0.2, name = "lrelu"):
    return tf.maximum(x, leak*x)
 
#定义relu激活层
def relu(input_, name = "relu"):
    return tf.nn.relu(input_, name = name)

def diydecay(steps,baselr):
    decay_steps = 100
    decay_rate=0.96
    cycle_step=100000
    n=steps//cycle_step
    clr=baselr*(0.8**n)
    
    steps=steps-n*cycle_step
    k=steps//decay_steps
    i=(-1)**k
    step=((i+1)/2)*steps-i*((k+1)//2)*decay_steps
     
    lr=tf.train.exponential_decay(clr,step,decay_steps,decay_rate, staircase=False)              
    return lr
def decay(global_steps,baselr):
    start_decay_step = 100000
    lr=tf.where(tf.greater_equal(global_steps,start_decay_step),
                diydecay(global_steps-start_decay_step,baselr),
                baselr)
    return lr 
def l2_loss(x):
    return tf.sqrt(tf.reduce_sum(x**2))
def grad(src):
    g_src_x = src[:, 1:, :, :] - src[:, :-1, :, :]
    g_src_y = src[:, :, 1:, :] - src[:, :, :-1, :]
    return g_src_x,g_src_y
def all_comp(grad1,grad2):
    v=[]
    dim1=grad1.get_shape()[-1]
    dim2=grad2.get_shape()[-1]
    for i in range(dim1):
        for j in range(dim2):
            v.append(tf.reduce_mean(((grad1[:,:,:,i]**2)*(grad2[:,:,:,j]**2)))**0.25)
    return v
def get_grad(src,dst,level):
    gradx_loss=[]
    grady_loss=[]
    for i in range(level):
        gradx1,grady1=grad(src)
        gradx2,grady2=grad(dst)
        # lambdax2=2.0*tf.reduce_mean(tf.abs(gradx1))/tf.reduce_mean(tf.abs(gradx2))
        # lambday2=2.0*tf.reduce_mean(tf.abs(grady1))/tf.reduce_mean(tf.abs(grady2))
        lambdax2=1
        lambday2=1
        gradx2_s=lambdax2*gradx2
        grady2_s=lambday2*grady2
        gradx_loss+=all_comp(gradx1,gradx2_s)
        grady_loss+=all_comp(grady1,grady2_s)
        src=avg_pooling(src,2,2,name='src_p'+str(i))
        dst=avg_pooling(dst,2,2,name='dst_p'+str(i))
    return gradx_loss,grady_loss

def exlusion_loss(src,dst,level=3):
    dim1=tf.cast(src.get_shape()[-1],dtype=tf.float32)
    dim2=tf.cast(dst.get_shape()[-1],dtype=tf.float32)
    gradx_loss,grady_loss=get_grad(src,dst,level)
    loss_gradxy=sum(gradx_loss)/(level*dim1*dim2)+sum(grady_loss)/(level*dim1*dim2)
    return loss_gradxy/2.0

def smooth_loss(src):
    x,y=grad(src)    
    g_loss=tf.reduce_mean(tf.abs(x))+tf.reduce_mean(tf.abs(y))
    return g_loss

def laplace_filter(src):
    i=src[:,1:-1,1:-1,:]
    left_i=src[:,:-2,:1:-1,:]
    right_i=src[:,2:,:1:-1,:]
    up_i=src[:,1:-1,:-2,:]
    down_i=src[:,1:-1,2:,:]
    output=4*i-(left_i+right_i+up_i+down_i)
    return output


