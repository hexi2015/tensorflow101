# coding=utf8
import tensorflow as tf

output_size = 5

input = tf.reshape(tf.constant([[1,2,3],
                                [4,5,6],
                                [7,8,9]], dtype=tf.float32), [1,3,3,1])
kernel = tf.reshape(tf.constant([[1,0,0],
                               [0,1,0],
                               [0,0,1]], dtype=tf.float32),[3,3,1,1])
# 计算过程
# 1. 根据步数 strides 对输入的内部进行填充，strides可以理解成输入放大的倍数，即在input的每个
#    元素之间填充0，0的个数n与strides的关系为：
#    n = strides-1
#    input_pad = [ [1,0,2,0,3],
#                  [0,0,0,0,0],
#                  [4,0,5,0,6],
#                  [0,0,0,0,0],
#                  [7,0,8,0,9] ]
#
#    因为卷积类型为 same，所以此时，i=5, k=3, s=1, p=1
# 2. 接下来, 用卷积核kernel 对填充后的输入 input_pad 进行步长 strides=1 的正向卷积
#    输出尺寸 o = (i-k+2p)/s + 1 = (5-3+2)/1 + 1 = 5
#    反卷积公式中我们给出的输出尺寸参数 output_shape 也是 5
#
#    input_pad_2 = [[0,0,0,0,0,0,0],
#                   [0,1,0,2,0,3,0],
#                   [0,0,0,0,0,0,0],
#                   [0,4,0,5,0,6,0],
#                   [0,0,0,0,0,0,0],
#                   [0,7,0,8,0,9,0],
#                   [0,0,0,0,0,0,0]]
#    output = [[1,0,2,0,3],
#              [0,6,0,2,0],
#              [4,0,5,0,6],
#              [0,12,0,14,0],
#              [7,0,8,0,9]]
tanspose_conv = tf.nn.conv2d_transpose(value=input,
                                       filter=kernel,
                                       output_shape=[1,output_size,output_size,1],
                                       strides=[1,2,2,1],
                                       padding='SAME')
out = tf.squeeze(tanspose_conv)

# 我们如果将output_shape 设置为4 是运行报错的，output_shape 只能等于5或者6
# 当我们将output_shape设置成 6 时, 发现input_pad_2 不能满足输出是6的需要，需要继续填充0
# tensorflow 优先在左侧和上侧填充0，填充后变为：
# #    input_pad_3 = [[0,0,0,0,0,0,0,0],
# #                   [0,0,0,0,0,0,0,0],
# #                   [0,0,1,0,2,0,3,0],
# #                   [0,0,0,0,0,0,0,0],
# #                   [0,0,4,0,5,0,6,0],
# #                   [0,0,0,0,0,0,0,0],
# #                   [0,0,7,0,8,0,9,0],
# #                   [0,0,0,0,0,0,0,0]]
#       output=     [[ 1.  0.  2.  0.  3.  0.]
#                    [ 0.  1.  0.  2.  0.  3.]
#                    [ 4.  0.  6.  0.  8.  0.]
#                    [ 0.  4.  0.  5.  0.  6.]
#                    [ 7.  0. 12.  0. 14.  0.]
#                    [ 0.  7.  0.  8.  0.  9.]]
tanspose_conv2 = tf.nn.conv2d_transpose(value=input,
                                       filter=kernel,
                                       output_shape=[1,output_size+1,output_size+1,1],
                                       strides=[1,2,2,1],
                                       padding='SAME')
out2 = tf.squeeze(tanspose_conv2)

with tf.Session() as sess:
    print sess.run(out)
    print sess.run(out2)

###################################################
# 反卷积只能恢复尺寸，不能恢复数值
# 重置图
tf.reset_default_graph()

value = tf.reshape(tf.constant([[1,2,3],
                                [4,5,6],
                                [7,8,9]],dtype=tf.float32), [1,3,3,1])
filter = tf.reshape(tf.constant([[1,0],
                                 [0,1]], dtype=tf.float32), [2,2,1,1])
conv = tf.nn.conv2d(value, filter, [1,1,1,1], 'SAME')
# output = [[ 6.  8.  3.]
#           [12. 14.  6.]
#           [ 7.  8.  9.]]
conv_output = tf.squeeze(conv)


conv_trans = tf.nn.conv2d_transpose(value = conv,
                                filter = filter,
                                output_shape=[1,3,3,1],
                                strides=[1,1,1,1],
                                padding='SAME'
                                )
conv_trans_output = tf.squeeze(conv_trans)
with tf.Session() as sess:
    # 卷积和反卷积后的结果并不一样
    print sess.run(conv_output)
    print sess.run(conv_trans_output)