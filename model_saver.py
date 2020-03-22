# coding=utf8
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
import numpy as np

def save_model():
    with tf.Graph().as_default() as g:
        v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
        v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
        result = v1 + v2


    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess.run(init)
        print sess.run(result)
        # 下面语句生成三个文件
        #  model.ckpt.meta: 计算图结构
        #  model.ckpt.index: 保存的是TensorFlow当前的变量名
        #  model.ckpt.data: 保存的是Tensorflow当前变量值
        saver.save(sess, "model/model.ckpt")

def restore_model():
    #############################
    # 定义一模一样的图
    with tf.Graph().as_default() as g:
        v1 = tf.Variable(tf.constant(0.0, shape=[1]), name="v1")
        v2 = tf.Variable(tf.constant(0.0, shape=[1]), name="v2")
        result = v1 + v2

    with tf.Session(graph=g) as sess:
        saver = tf.train.Saver()
        # 不需要运行变量初始化，从模型文件中加载变量的值
        saver.restore(sess, "model/model.ckpt")
        # 结果应该是 3
        print sess.run(result)

def import_model():
    """
    前面说了很多关于加载变量，下面说一说如何加载模型。
    如果不希望在加载模型的时候重复定义计算图，可以直接加载已经持久化的图。
    对于加载模型的操作TensorFlow也提供了很方便的函数调用，
    我们还记得保存模型时候将计算图保存到.meta后缀的文件中。那此时只需要加载这个文件即可：

    """
    with tf.Graph().as_default() as g:
        # 加载持久化的图
        saver = tf.train.import_meta_graph("model/model.ckpt.meta")

    with tf.Session(graph=g) as sess:
        # 加载变量
        saver.restore(sess, "model/model.ckpt")

        # 通过张量名称获取张量
        print sess.run(g.get_tensor_by_name("add:0"))
        print(sess.run(g.get_tensor_by_name("v1:0")))
        print(sess.run(g.get_tensor_by_name("v2:0")))

def restore_model_part():
    """
    我们在保存模型的时候知道，在保存模型的时候，我们可以给tf.train.Saver()中传递参数实现一些高级的实现，比如：

    1. 参数指定一个列表，指定部分变量进行保存，列表中的元素是变量名；
    2. 参数指定一个变量名与变量名称对应的字典来指定保存时候的对应关系，
       因为此时保存的时候和变量名没有关系了，而是以变量名称作为唯一的标识；
    保存的时候可以这样指定，其实在加载模型的时候，同样可以这样操作

    """
    #############################
    # 定义一模一样的图
    with tf.Graph().as_default() as g:
        v1 = tf.Variable(tf.constant(0.0, shape=[1]), name="v1")
        v2 = tf.Variable(tf.constant(0.0, shape=[1]), name="v2")
        result = v1 + v2

    with tf.Session(graph=g) as sess:
        #saver = tf.train.Saver([v1])
        saver = tf.train.Saver({"v1":v1})
        # 初始化必须加上，否则会报错，因为v2没有被初始化
        sess.run(tf.global_variables_initializer())
        # 不需要运行变量初始化，从模型文件中加载变量的值
        saver.restore(sess, "model/model.ckpt")
        # 结果应该是 1
        print "result=", sess.run(result)
        print "v1=", sess.run(v1)
        print "v2=", sess.run(v2)

def restore_model_alias():
    #############################
    # 这里声明的变量名称和已经保存的模型中的变量名称不同
    with tf.Graph().as_default() as g:
        v1 = tf.Variable(tf.constant(0.0, shape=[1]), name="other-v1")
        v2 = tf.Variable(tf.constant(0.0, shape=[1]), name="other-v2")
        result = v1 + v2

    with tf.Session(graph=g) as sess:
        # 直接用 saver = tf.train.Saver()，会报错，因为模型中没有key=other-v1 的变量
        # 字典指明模型文件中 名称为"v1"的变量加载到 变量v1（名字是other-v1），名称为"v2"的变量加载到 变量v2(名字是other-v2)，
        saver = tf.train.Saver({"v1":v1, "v2":v2})
        # 不需要运行变量初始化，从模型文件中加载变量的值
        saver.restore(sess, "model/model.ckpt")
        # 结果应该是 3
        print sess.run(result)


def save_graph_pb():
    with tf.Graph().as_default() as g:
        v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
        v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
        result = v1 + v2

    with tf.Session(graph=g) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # 导出当前图的GraphDef部分, 只需要这一部分就可以完成从输入层到输出层的计算过程
        graph_def = g.as_graph_def()

        # convert_variables_to_constants：通过这个函数可以将计算图中的变量及其取值通过常量保存。
        # add 没有“:0”，表示这是计算节点，而“add:0” 表示节点计算后的输出张量
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['add'])
        with tf.gfile.GFile("model/combined_model.pb", 'wb') as f:
            f.write(output_graph_def.SerializeToString())

def load_graph_pb():
    g = tf.Graph()
    with tf.Session(graph=g) as sess:
        model_filename = "model/combined_model.pb"
        with gfile.FastGFile(model_filename,"rb") as f:
            graph_def = g.as_graph_def()
            graph_def.ParseFromString(f.read())

        # 将graph_def 中保存的图加载到当前的图中，return_elements = ['add:0'] 给出了返回的张量的名称。
        # 在保存的时候给出的是计算节点的名称，所以为"add"。在加载的时候给出的是张量的名称，所以是 "add:0"

        result = tf.import_graph_def(graph_def, return_elements=["add:0"])
        print sess.run(result)


save_model()
restore_model()
import_model()
restore_model_part()
restore_model_alias()
save_graph_pb()
load_graph_pb()