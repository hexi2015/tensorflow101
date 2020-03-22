# coding=utf8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def _int64_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def write_tfrecord(filename):
    mnist = input_data.read_data_sets("MNIST_data", dtype=tf.uint8, one_hot=True)
    images = mnist.train.images
    labels = mnist.train.labels
    pixels = images.shape[1]
    num_examples = mnist.train.num_examples

    writer = tf.python_io.TFRecordWriter(filename)
    for index in range(num_examples):
        # 将图片转化成一个字符串
        image_raw = images[index].tostring()
        label_onehot = [0]*10
        label_onehot[np.argmax(labels[index])] = 1
        # 将一个样例转化为 Example protocol buffer，并将所有信息写入这个数据结构
        example = tf.train.Example(features=tf.train.Features(feature={
            'pixels': _int64_feature(pixels),
            'label': _int64_feature(np.argmax(labels[index])),
            'label_onehot': _int64_feature(label_onehot),
            'image_raw': _bytes_feature(image_raw)
        }))
        writer.write(example.SerializeToString())
    writer.close()

def read_tfrecord(filename):
    # 创建一个reader来读取TFRecord 文件中的样例
    reader = tf.TFRecordReader()
    # 创建一个队列来维护输入文件列表
    filename_queue = tf.train.string_input_producer([filename])
    # 从文件中读取一个样例, 也可以使用 read_up_to 一次性读取多个样例
    _, serialized_example = reader.read(filename_queue)
    # 解析读入的一个样例，如果需要解析多个样例，可以用parse_example
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'pixels': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'label_onehot': tf.FixedLenFeature([10],tf.int64)
        }
    )
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    label = tf.cast(features['label'], tf.int32)
    pixels = tf.cast(features['pixels'], tf.int32)
    label_onehot = tf.cast(features['label_onehot'], tf.int32)

    sess = tf.Session()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for i in range(10):
        _, label_np, label_onehot_np, pixels_np = sess.run([image, label, label_onehot, pixels])
        print label_np, label_onehot_np

    sess.close()


filename = 'mnist.record'
write_tfrecord(filename)
read_tfrecord(filename)