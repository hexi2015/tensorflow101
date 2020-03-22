# coding=utf8
import tensorflow as tf


# 定义一个变量用于计算滑动平均
v1 = tf.Variable(0, dtype=tf.float32)

# 模拟神经网络迭代次数，动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类。初始给定了 decay=0.99， 控制衰减率的变量step；

ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时
# 这个列表中的变量就会被更新

maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过 ema.average(v1) 获取滑动平均之后的变量的取值
    # 初始化后，变量v1 和 滑动平均都是 0
    print sess.run([v1, ema.average(v1)])

    # 更新变量v1的值为5
    sess.run(tf.assign(v1,5))
    # 更新v1 的滑动平均值。衰减率为 min(0.99, (1+step)/(10+step)=0.1) = 0.1
    # 所以，v1的滑动平均值会被更新为 0*0.1+5*0.9 = 4.5
    sess.run(maintain_averages_op)
    # [5.0, 4.5]
    print sess.run([v1, ema.average((v1))])

    # 更新step=10000
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))

    # 更新v1 的滑动平均值。衰减率为 min(0.99, (1+step)/(10+step)=0.999) =0.99
    # 所以v1 的滑动平均值会被更新为 4.5*0.99 + 10 * 0.01 = 4.555

    sess.run(maintain_averages_op)
    # [10.0, 4.555]
    print sess.run([v1, ema.average(v1)])

    # 再次更新滑动平均值，得到的新滑动平均值为 0.99 * 4.555 + 10*0.01 = 4.60945
    sess.run(maintain_averages_op)
    print sess.run([v1, ema.average(v1)])