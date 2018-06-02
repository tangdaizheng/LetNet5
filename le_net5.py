#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf


def get_variable(name, shape, initializer, regularizer=None, dtype='float', trainable=True):
    collections = [tf.GraphKeys.GLOBAL_VARIABLES]
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           regularizer=regularizer,
                           collections=collections,
                           dtype=dtype,
                           trainable=trainable)


def conv2d(x, ksize, stride, filter_out, name, padding):
    with tf.variable_scope(name):
        filter_in = x.get_shape()[-1]
        stddev = 1. / tf.sqrt(tf.cast(filter_out, tf.float32))
        weight_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        bias_initializer = tf.random_uniform_initializer(minval=-stddev, maxval=stddev, dtype=tf.float32)
        shape = [ksize, ksize, filter_in, filter_out]
        kernel = get_variable('kernel', shape, weight_initializer)
        bias = get_variable('bias', [filter_out], bias_initializer)
        conv = tf.nn.conv2d(x, kernel, [1, stride, stride, 1], padding=padding)
        out = tf.nn.bias_add(conv, bias)
        return tf.nn.relu(out)

def max_pool(x, ksize, stride, name, padding):
    return tf.nn.max_pool(x, [1, ksize, ksize, 1], [1, stride, stride, 1], name=name, padding=padding)


def flatten(x):
    shape = x.get_shape().as_list()
    dim = 1
    for i in range(1, len(shape)):
        dim *= shape[i]
    return tf.reshape(x, [-1, dim]), dim

def fc_layer(x, i_size, o_size, name, is_relu=None):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable('w', shape=[i_size, o_size], dtype='float')
        b = tf.get_variable('b', shape=[o_size], dtype='float')
        out = tf.nn.xw_plus_b(x, w, b, name=scope.name)
        if is_relu:
            out = tf.nn.relu(out)
        return out

def drop_out(x, keep_prob, name):
    return tf.nn.dropout(x, keep_prob=keep_prob, name=name)


class LetNet5(object):

    def __init__(self, x, n_class=10, keep_prob=1.0):
        self.input = x
        self.n_class = n_class
        self.keep_prob = keep_prob

        self._build_net()

    def _build_net(self):
        with tf.name_scope('conv_1'):
            conv1 = conv2d(self.input, 5, 1, 6, 'conv1', 'VALID')
        with tf.name_scope('pool_1'):
            pool1 = max_pool(conv1, 1, 1, 'pool1', 'VALID')

        with tf.name_scope('conv_2'):
            conv2 = conv2d(pool1, 5, 1, 16, 'conv2', 'VALID')
        with tf.name_scope('pool_2'):
            pool2 = max_pool(conv2, 1, 1, 'pool2', 'VALID')

        with tf.name_scope('conv_3'):
            conv3 = conv2d(pool2, 5, 1, 120, 'conv3', 'VALID')

        with tf.name_scope('flat_1'):
            flat1, flat_dim = flatten(conv3)

        with tf.name_scope('fc_1'):
            fc1 = fc_layer(flat1, flat_dim, 84, 'fc1')

        with tf.name_scope('fc_2'):
            fc2 = fc_layer(fc1, 84, 10, 'fc2')

        with tf.name_scope('drop_out_1'):
            drop1 = drop_out(fc2, self.keep_prob, 'drop_out1')
            self.logits = drop1

        with tf.name_scope('prediction'):
            self.prediction = tf.nn.softmax(self.logits)
