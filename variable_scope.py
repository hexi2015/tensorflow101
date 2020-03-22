# coding=utf8
import tensorflow as tf
import numpy as np

########################################
#  Variable
#  tf.get_variable 的好处，

# 推荐使用tf.get_variable(), 因为：

# 1. 初始化更方便,比如用xavier_initializer:
#    W = tf.get_variable("W", shape=[784, 256],
#    initializer=tf.contrib.layers.xavier_initializer())

# 2. 方便共享变量
#    因为tf.get_variable() 会检查当前命名空间下是否存在同样name的变量，可以方便共享变量。而tf.Variable 每次都会新建一个变量。
#    需要注意的是tf.get_variable() 要配合reuse和tf.variable_scope() 使用。

########################################

# variable_scope 基本用法
# 在构建大型网络结构时，经常用variable_scope 定义一个模块

with tf.Graph().as_default() as g1:
    # x为训练图像的占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32, [None, 784])
    #y_ = tf.placeholder(tf.float32, [None, 10])

    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    # variable_scope 定义一个操作block，block内部变量和操作的命名都会以 "conv1" 为前缀
    with tf.variable_scope("conv1") as scope:
	    w_conv = tf.get_variable(name="w",
							 shape=[3,3,1,32],
							 dtype=tf.float32,
							 initializer=tf.truncated_normal_initializer(stddev=0.1))
	    b_conv = tf.get_variable(name="b",
							 shape=[32],
							 dtype=tf.float32,
							 initializer=tf.constant_initializer(0.1)
							 )
	    h_conv = tf.nn.conv2d(x_image, w_conv, strides=[1, 1, 1, 1], padding='SAME')
	    pre_activation = tf.nn.bias_add(h_conv, b_conv)
	    conv = tf.nn.relu(pre_activation, name=scope.name)
	    pool = tf.nn.max_pool(conv,
							ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')

    # 输出所有变量的名字
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print "%d variables created:" % len(vars)
    for var in vars:
        print var.name

    # 输出所有操作的名字
    ops = g1.get_operations()
    print "\n%d operations:" % len(ops)
    for op in ops:
        print op.name


def conv_block(name, input, kernel, num_filter, reuse=False):
    # variable_scope 定义一个操作block，block内部变量和操作的命名都会以 name 为前缀
    channels = input.get_shape()[3].value
    with tf.variable_scope(name, reuse=reuse) as scope:
        w_conv = tf.get_variable(name="w",
                                 shape=[kernel, kernel, channels, num_filter],
                                 dtype=tf.float32,
                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        b_conv = tf.get_variable(name="b",
                                 shape=[num_filter],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1)
                                 )
        h_conv = tf.nn.conv2d(input, w_conv, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(h_conv, b_conv)
        conv = tf.nn.relu(pre_activation, name=scope.name)
        pool = tf.nn.max_pool(conv,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    return pool

def build_network1(input):
    conv1 = conv_block("conv1", input, 3, 32)
    conv2 = conv_block("conv2", conv1, 3, 64)

    size = 7*7
    with tf.variable_scope("fc1") as scope:
        W_fc = tf.get_variable(name="w",
                                shape=[size,10],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(stddev=0.1))

        b_fc = tf.get_variable(name="b", shape=[10,],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0.1))
        h_conv_flat = tf.reshape(conv2, [-1, size])
        h_fc = tf.nn.relu(tf.matmul(h_conv_flat, W_fc) + b_fc)

    return h_fc


def build_network2(input):
    # 共享网络1 的卷积操作，如果不共享，name 必须不能重复
    conv1 = conv_block("conv1", input, 3, 32, reuse=True)
    conv2 = conv_block("conv2", conv1, 3, 64, reuse=True)

    size = 7 * 7
    with tf.variable_scope("fc2") as scope:
        W_fc = tf.get_variable(name="w",
                            shape=[size, 10],
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))

        b_fc = tf.get_variable(name="b", shape=[10, ],
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        h_conv_flat = tf.reshape(conv2, [-1, size])
        h_fc = tf.nn.relu(tf.matmul(h_conv_flat, W_fc) + b_fc)

    return h_fc

with tf.Graph().as_default() as g2:
    # x为训练图像的占位符、y_为训练图像标签的占位符
    x = tf.placeholder(tf.float32, [None, 784])
    # y_ = tf.placeholder(tf.float32, [None, 10])
    # 将单张图片从784维向量重新还原为28x28的矩阵图片
    x_image = tf.reshape(x, [-1, 28, 28, 1])

    y1 = build_network1(x_image)
    y2 = build_network2(x_image)

    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    print "%d variables created:" % len(vars)
    for var in vars:
        print var.name
    # 结果是：
    # conv1 / w:0
    # conv1 / b:0
    # conv2 / w:0
    # conv2 / b:0
    # fc1 / w:0
    # fc1 / b:0
    # fc2 / w:0
    # fc2 / b:0

    # conv1, conv2 的变量只被创建一次

