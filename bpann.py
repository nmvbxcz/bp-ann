#!/usr/bin/python
#  -*- coding: UTF-8 -*-
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


filename_queue = tf.train.string_input_producer(["/home/my_admin/winequality-white.csv"])
reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)

xs = tf.placeholder(tf.float32, [None, 11])
ys = tf.placeholder(tf.float32, [None, 1])

def weight_variable(shape):
  initial=tf.truncated_normal(shape,stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial=tf.constant(0.1,shape=shape)
  return tf.Variable(initial)

def add_layer(inputs, weights, biases, activation_function=None):
    # 构建权重 : in_size * out)_sieze 大小的矩阵
    # weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # 构建偏置 : 1 * out_size 的矩阵
    # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # 矩阵相乘
    Wx_plus_b = tf.matmul(inputs, weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs  # 得到输出数据

def compute_accuracy(v_xs, v_ys):
  global prediction,h1
  h1_pre=sess.run(h1,feed_dict={xs:v_xs})
  print(h1_pre)
  y_pre=sess.run(prediction,feed_dict={xs:v_xs})
  # print(v_ys,y_pre)
  # return y_pre


hidden_layers = 10

h1_wights=weight_variable([11,hidden_layers])
h1_b=bias_variable([1,hidden_layers])
prediction_wight=weight_variable([hidden_layers,1])
prediction_b=bias_variable([1,1])

# 构建输入层到隐藏层,假设隐藏层有 hidden_layers 个神经元
h1 = add_layer(xs, h1_wights, h1_b, activation_function=tf.nn.relu)
# 构建隐藏层到隐藏层
# h2 = add_layer(h1, hidden_layers, hidden_layers, activation_function=tf.nn.relu)
# # 构建隐藏层到隐藏层
# h3 = add_layer(h2, hidden_layers, hidden_layers, activation_function=tf.nn.sigmoid)
# 构建隐藏层到输出层
prediction = add_layer(h1, prediction_wight, prediction_b, activation_function=None)

# 接下来构建损失函数: 计算输出层的预测值和真是值间的误差,对于两者差的平方求和,再取平均,得到损失函数.运用梯度下降法,以0.1的效率最小化损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss) # 优化算法选取SGD,随机梯度下降

record_defaults = [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]]
col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12 = tf.decode_csv(value, record_defaults=record_defaults,field_delim=';')

features = tf.stack([col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11])

x_data = np.empty(shape=[0, 11])
y_data = np.empty(shape=[0, 1])
x_test = np.empty(shape=[0, 11],dtype=tf.float32)
y_test = np.empty(shape=[0, 1],dtype=tf.float32)

init_op = tf.global_variables_initializer()
local_init_op = tf.local_variables_initializer()

with tf.Session() as sess:
  sess.run(init_op)
  sess.run(local_init_op)

  #save to path
  #save_path = saver.save(sess, "my_net/save_net.ckpt")

  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  try:
    for i in range(3898):
      example, label = sess.run([features, col12])
      # print(label)
      x_data = np.row_stack((x_data,example))
      y_data = np.row_stack((y_data,label))
    for j in range(1000):
      example, label = sess.run([features, col12])
      # print(label)
      x_test = np.row_stack((x_test,example))
      y_test = np.row_stack((y_test,label))
  except tf.errors.OutOfRangeError:
    print 'Done !!!'

  finally:
    coord.request_stop()
    coord.join(threads)
  # print y_test

  for k in range(1000):
      sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
      # print(' loss is : %i', sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
  compute_accuracy(x_test,y_test)
  # print(h1_wights,h1_b,prediction_wight,prediction_b)
  # print(sess.run(h1_wights),sess.run(h1_b))
  res1=tf.matmul(x_test, sess.run(h1_wights))
  res2=tf.cast(sess.run(h1_b))