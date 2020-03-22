# coding=utf8
import tensorflow as tf
import numpy as np

########################################
#  如何定义 Const
########################################
with tf.Graph().as_default() as g1:
    # python list 初始化，无需指定shape
    a = tf.constant([1.0]*64, dtype=tf.float32, name='bias')
    # 指定 value 和 shape，初始化为 value
    b = tf.constant(1.0, dtype=tf.float32,shape=[64], name='bias')
    # 从 numpy 初始化
    c = tf.constant(np.ones((64,)), dtype=tf.float32,  name="bias")
    # tf 支持的预定义的tensor
    d = tf.ones((64,), dtype=tf.float32, name='bias')

    print a
    print b
    print c
    print d

    o = a+b+c+d

with tf.Session(graph=g1) as sess:
    r = sess.run(o)
    print r

########################################
#  如何定义 Variable
########################################
with tf.Graph().as_default() as g2:
    x = tf.random_uniform((1,64), minval=-1.0, maxval=1.0)
    # 通过初始值tensor来定义variable，并初始化
    w1 = tf.Variable(tf.truncated_normal([64,10], stddev=0.1))
    b1 = tf.Variable(tf.constant(0.1, shape=[10,]))

    # tf.constant_initializer：常量初始化函数
    # tf.random_normal_initializer：正态分布
    # tf.truncated_normal_initializer：截取的正态分布
    # tf.random_uniform_initializer：均匀分布
    # tf.zeros_initializer：全部是0
    # tf.ones_initializer：全是1
    # tf.uniform_unit_scaling_initializer：满足均匀分布，但不影响输出数量级的随机值

    # 指定name，shape 和 initializer来初始化
    w2 = tf.get_variable(name="w2", shape=[64,10],
                         dtype=tf.float32,
                         initializer=tf.truncated_normal_initializer(stddev=0.1))
    b2 = tf.get_variable(name="b2", shape=[10,],
                         dtype=tf.float32,
                         initializer=tf.constant_initializer(0.1))

    o = tf.nn.relu(tf.matmul(x,w1)+b1) + tf.nn.relu(tf.matmul(x,w2)+b2)

    print "w1",w1
    print "b1",b1
    print "w2",w2
    print "b2",b2


with tf.Session(graph=g2) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    r = sess.run(o)
    print r

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