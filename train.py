#!/usr/bin/python3
# -*- coding:utf-8 -*-

import os
import tensorflow as tf
from le_net5 import LetNet5
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_integer('epoch', 20000, 'epoch')
tf.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.flags.DEFINE_float('lr', 0.001, 'learning rate')
tf.flags.DEFINE_boolean('restore', False, 'restore from checkpoint and run test')


data_dir = 'mnist_data'
ckpt_dir = 'ckpt/'

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x_input')
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_ = tf.placeholder(tf.float32, [None, 10], name='label_input')

with tf.name_scope('prediction'):
    le_net5 = LetNet5(x_image, keep_prob)
    logit = le_net5.logits
    y = le_net5.prediction

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y)))

# with tf.name_scope('loss'):
#     print('logit shape', logit.get_shape(), 'label shape:', y_.get_shape())
#     logit_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=tf.argmax(y_, 1)))
#     l2_loss = tf.contrib.layers.apply_regularization(regularizer=tf.contrib.layers.l2_regularizer(1e-5),
#                                                      weights_list=tf.trainable_variables())
#     loss = logit_loss + l2_loss
    
with tf.name_scope('train_step'):
    train_step = tf.train.AdagradOptimizer(FLAGS.lr).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        mnist = input_data.read_data_sets(data_dir, one_hot=True)

        for i in range(FLAGS.epoch):
            if FLAGS.restore:
                saver.restore(sess, tf.train.latest_checkpoint(ckpt_dir))
            else:
                train_image_batch, train_label_batch = mnist.train.next_batch(FLAGS.batch_size)
                sess.run(train_step, feed_dict={x: train_image_batch, y_: train_label_batch, keep_prob: 0.5})

            if i % 10 == 0 or FLAGS.restore:
                test_image_batch, test_label_batch = mnist.test.next_batch(FLAGS.batch_size)
                print('iter:' + str(i), sess.run(accuracy, feed_dict={x: test_image_batch, y_: test_label_batch, keep_prob: 1.0}))

            if i > 0 and i % 100 == 0 and not FLAGS.restore:  #  保存checkpoint
                saver.save(sess, ckpt_dir, global_step=i)




