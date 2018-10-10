# 描述网络结构
import tensorflow as tf

INPUT_NODE = 784  # 输入节点28*28
OUTPUT_NODE = 10  # 输出索引号出现的概率
LAYER1_NODE = 1000  # 隐藏层节点个数


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))  # 随机生成w
    if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(
        w))  # 将每个变量的正则化损失加入损失集合losses
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):  # 描述从输入到输出的数据流
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)  # 第一层的参数、偏置、输出

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])
    y = tf.matmul(y1, w2) + b2  # 第二层的参数、偏置、输出。其中，y直接输出，因为要对输出使用softmax函数使其符合概率分布，故不经过relu函数
    return y
