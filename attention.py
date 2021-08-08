# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:06:33 2018

@author: Neoooli
"""

import tensorflow as tf
from allnet import *

def pattentionnet(image, gf_dim=32, reuse=False, name="generator"):
    # dropout_rate = 0.8
    output_dim=image.get_shape()[-1]
    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        # image is (256 x 256 x input_c_dim)
        e0 = conv2d(image,gf_dim,stride=1,name='g_e0_conv')
        # e1 is (128 x 128 x self.gf_dim)
        e1 = conv2d(lrelu(instance_norm(e0,'g_bn_e1')), gf_dim*2, name='g_e1_conv')
        e2 = conv2d(lrelu(instance_norm(e1, 'g_bn_e2')), gf_dim*4, name='g_e2_conv')
        # e2 is (64 x 64 x self.gf_dim*2)
        e3 = conv2d(lrelu(instance_norm(e2, 'g_bn_e3')), gf_dim*8, name='g_e3_conv')
        # e3 is (32 x 32 x self.gf_dim*4)
        e4 = conv2d(lrelu(instance_norm(e3, 'g_bn_e4')), gf_dim*16, name='g_e4_conv')
        # e4 is (16 x 16 x self.gf_dim*8)
        e5 = conv2d(lrelu(instance_norm(e4, 'g_bn_e5')), gf_dim*16, name='g_e5_conv')
        # e5 is (8 x 8 x self.gf_dim*8)
        e6 = conv2d(lrelu(instance_norm(e5, 'g_bn_e6')), gf_dim*16, name='g_e6_conv')
        # e6 is (4 x 4 x self.gf_dim*8)
      
        d1 = deconv2d(tf.nn.relu(instance_norm(e6,'d_bn_d0')), gf_dim*16, name='g_d1')
#        d1 = tf.nn.dropout(d1, dropout_rate)
        d1 = instance_norm(tf.concat([d1, e5],3),'g_bn_d1')
        # d1 is (8 x 8 x self.gf_dim*8*2)

        d2 = deconv2d(tf.nn.relu(d1), gf_dim*16, name='g_d2')
#        d2 = tf.nn.dropout(d2, dropout_rate)
        d2 = instance_norm(tf.concat([d2,e4], 3),'g_bn_d2')
        # d2 is (16 x 16 x self.gf_dim*8*2)

        d3 = deconv2d(tf.nn.relu(d2), gf_dim*8, name='g_d3')
#        d3 = tf.nn.dropout(d3, dropout_rate)
        d3 = instance_norm(tf.concat([d3, e3], 3),'g_bn_d3')
        # d3 is (32 x 32 x self.gf_dim*8*2)

        d4 = deconv2d(tf.nn.relu(d3), gf_dim*4, name='g_d4')
        d4 = instance_norm(tf.concat([d4,e2], 3),'g_bn_d4')
        # d4 is (16 x 16 x self.gf_dim*8*2)

        d5 = deconv2d(tf.nn.relu(d4), gf_dim*2, name='g_d5')
        d5 = instance_norm(tf.concat([d5,e1], 3),'g_bn_d5')
        # d5 is (32 x 32 x self.gf_dim*4*2)

        d6 = deconv2d(tf.nn.relu(d5),gf_dim,name='g_d6')
        d6 = instance_norm(tf.concat([d6,e0],axis=3),'g_bn_d6')
        # d8 is (256 x 256 x output_c_dim)
        d6 = conv2d(tf.nn.relu(d6),output_dim*3, kernel_size = 1, stride = 1,name = 'out_conv')
        t=tf.ones_like(image[:,:,:,0:1])
        r=tf.ones_like(image[:,:,:,0:1])

        for i in range(output_dim):
            b=tf.nn.softmax(d6[:,:,:,i*3:3+i*3])
            t=tf.concat([t,b[:,:,:,0:1]],axis=3)
            r=tf.concat([r,b[:,:,:,1:2]],axis=3)
        return d6,t[:,:,:,1:],r[:,:,:,1:]