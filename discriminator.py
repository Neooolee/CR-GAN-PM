# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 16:58:01 2018

@author: lijun
"""
import tensorflow as tf
from allnet import *
import denseblocks as db
import sys
sys.path.append('E:\lijun\packages\CliqueNet-master\models')
import cliquenet_I_II as cliquenet
#import resnet as resnet
def cnodiscriminator(image,reuse=False, name="discriminator"):
    df_dim=64
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(input_=image,output_dim=df_dim,kernel_size=4,stride=2,name='layer1-conv2d',biased=True))
        # h0 is (128 x 128 x self.df_dim)
        # h0 = lrelu(instance_norm(conv2d(input_=image,output_dim=df_dim,kernel_size=4,stride=2,name='layer1-conv2d',biased=True), 'd_bn0'))
        h1 = lrelu(instance_norm(conv2d(input_=h0,output_dim=df_dim*2,kernel_size=4,stride=2,name='layer2-conv2d',biased=True), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(input_=h1,output_dim=df_dim*4,kernel_size=4,stride=2,name='layer3-conv2d',biased=True), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(input_=h2,output_dim=df_dim*8,kernel_size=4,stride=2,name='layer4-conv2d',biased=True), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)
        h4 = conv2d(input_=h3,output_dim=1,kernel_size=4,stride=1,name='layer5-conv2d',biased=True)
        # h4 is (32 x 32 x 1)
#        w=h4.get_shape().as_list()[1]
#        h4=avg_pooling(input_=h4, kernel_size=w, stride=1,name='global_avg_pool')
#        h4=tf.reduce_mean(h4,[1,2])
#        h4=tf.contrib.layers.flatten(h4)
#        h4= tf.layers.dense(inputs=h4, 
#                    units=1, 
#                    activation=tf.sigmoid,
#                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
#                    kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
        #h4=tf.sigmoid(h4)
        return h4
