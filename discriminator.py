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
def anodiscriminator(image,reuse=False,name='discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
#    输出维度1*256*256*64
        conv1=lrelu(conv2d(input_=image,output_dim=64,kernel_size=3,stride=1,name='layer1-conv2d',biased=True))
        conv2=lrelu(instance_norm(conv2d(input_=conv1,output_dim=128,kernel_size=3,stride=2,name='layer2-conv2d',biased=True), 'd_bn2'))

        conv3=lrelu(instance_norm(conv2d(input_=conv2,output_dim=128,kernel_size=3,stride=1,name='layer3-conv2d',biased=True), 'd_bn3'))
        conv4=lrelu(instance_norm(conv2d(input_=conv3,output_dim=256,kernel_size=3,stride=2,name='layer4-conv2d',biased=True), 'd_bn4'))
        conv5=lrelu(instance_norm(conv2d(input_=conv4,output_dim=256,kernel_size=3,stride=1,name='layer5-conv2d',biased=True), 'd_bn5'))
        conv6=lrelu(instance_norm(conv2d(input_=conv5,output_dim=512,kernel_size=3,stride=2,name='layer6-conv2d',biased=True), 'd_bn6'))
        conv7=lrelu(instance_norm(conv2d(input_=conv6,output_dim=512,kernel_size=3,stride=1,name='layer7-conv2d',biased=True), 'd_bn7'))
        conv8=lrelu(instance_norm(conv2d(input_=conv7,output_dim=512,kernel_size=3,stride=2,name='layer8-conv2d',biased=True), 'd_bn8'))
        conv9=lrelu(conv2d(input_=conv8,output_dim=1024,kernel_size=3,stride=2,name='layer9-conv2d',biased=True))

#        pool_shape=conv9.get_shape().as_list()
##    pool_shape[0]为一个batch中数据个数
#        nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
#        reshaped=tf.reshape(conv9,[pool_shape[0],nodes])
#        w=conv9.get_shape().as_list()[1]
#        p=avg_pooling(input_=conv9, kernel_size=w, stride=1,padding='VALID',name='global_avg_pool')
        logits= tf.layers.dense(inputs=conv9, 
                        units=1, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

        return logits

def bnodiscriminator(image,reuse=False,name='discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
#    输出维度1*256*256*64
        conv1=lrelu(conv2d(input_=image,output_dim=64,kernel_size=3,stride=1,name='layer1-conv2d',biased=True))
        conv2=lrelu(conv2d(input_=conv1,output_dim=64,kernel_size=3,stride=2,name='layer2-conv2d',biased=True))
        
        r1=residule_block(input_=conv2, output_dim=64,name = "res1")
        
        conv3=lrelu(conv2d(input_=r1,output_dim=128,kernel_size=3,stride=2,name='layer3-conv2d',biased=True))
        r2=residule_block(input_=conv3, output_dim=128,name = "res2")
        
        conv4=lrelu(conv2d(input_=r2,output_dim=256,kernel_size=3,stride=2,name='layer4-conv2d',biased=True))
        r3=residule_block(input_=conv4,output_dim=256, name = "res3")
        
        conv5=lrelu(conv2d(input_=r3,output_dim=512,kernel_size=3,stride=2,name='layer5-conv2d',biased=True))
        r4=residule_block(input_=conv5, output_dim=512, name = "res4")
        
        conv6=lrelu(conv2d(input_=r4,output_dim=512,kernel_size=3,stride=1,name='layer6-conv2d',biased=True))
        
        conv7=lrelu(conv2d(input_=conv6,output_dim=512,kernel_size=3,stride=2,name='layer7-conv2d',biased=True))


        pool_shape=conv7.get_shape().as_list()
#    pool_shape[0]为一个batch中数据个数
        nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
        reshaped=tf.reshape(conv7,[pool_shape[0],nodes])

        logits= tf.layers.dense(inputs=reshaped, 
                        units=1, 
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

        return logits

def nodiscriminator(image,reuse=False,name='densenet_discriminator'):
#    dropout_rate在这里指随机丢掉神经元的比例
    dropout_rate=0.0
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
        denseblock=db.DenseNet(n_class=1,nb_blocks=[4,4,4,4],dropout_rate=dropout_rate,n_filters_first=64,growth_rate=32)
        logits=denseblock(input_=image)
        logits=tf.sigmoid(logits)
        return logits
    
def dnodiscriminator(image,reuse=False,name='cliquenet_discriminator'):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False
#        keep_prob是随机保存节点比例，if_a是是否使用attention,if_b是否使用block,if_c是否压缩
        logits=cliquenet.build_model(input_=image,n_class=1,nb_blocks=[5,5,5,5],growth_rate=12,keep_prob=0.5,is_train=True,if_a=True,if_b=False,if_c=True)
        
        return logits
