# 描述网络参数的优化方法
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import os

BATCH_SIZE = 200  # 每轮喂入神经网络的图片数
LEARNING_RATE_BASE = 0.01  # 最开始的学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
REGULARIZER = 0.001  # 正则化系数
STEPS = 50000  # 共训练多少轮
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率
MODEL_SAVE_PATH = "./model/"  # 模型的保存路径
MODEL_NAME = "mnist_model"  # 模型保存的文件名


def backward(mnist):  # 读入mnist

    x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])  # placeholder占位
    y = mnist_forward.forward(x, REGULARIZER)  # 调用前向传播的程序计算输出y
    global_step = tf.Variable(0, trainable=False)  # 给轮数计数器赋初值，设定为不可训练

    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))  # 调用包含正则化的损失函数loss

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)  # 定义指数衰减学习率

    # train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)  # 定义训练过程
    # train_step = tf.train.MomentumOptimizer(learning_rate, momentum= 0.9).minimize(loss,global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')  # 定义滑动平均

    saver = tf.train.Saver()  # 实例化saver

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)  # 初始化所有变量

        for i in range(STEPS):  # 训练STEPS轮
            xs, ys = mnist.train.next_batch(BATCH_SIZE)  # 每次读入BATCH_SIZE组图片内容及其标签
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})  # 喂入神经网络并执行训练过程
            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))  # 每1000轮打印出loss值
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)  # 保存模型到当前会话


def main():
    mnist = input_data.read_data_sets("./data", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()
