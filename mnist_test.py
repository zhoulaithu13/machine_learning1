# 复现节点，计算测试集的准确率
import time  # 延迟
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_forward
import mnist_backward

TEST_INTERVAL_SECS = 5  # 定义程序循环的间隔时间是5s


def test():  # 先读入mnist数据集
    mnist = input_data.read_data_sets("./data/", one_hot=True)  # 读入数据集
    # sess = tf.InteractiveSession()
    with tf.Graph().as_default() as g:  # tf.Graph复现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])  # 占位
        y = mnist_forward.forward(x, None)  # 前向传播过程计算y的值

        ema = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)  # 实例化带滑动平均的saver对象，这样所有参数在会话中被加载时，会被赋值为各自的滑动平均值

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算准确率

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)  # 把滑动平均值赋给各个参数
                if ckpt and ckpt.model_checkpoint_path:  # 先判断是否已有模型，若有则恢复
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]  # 恢复global_step值
                    accuracy_score = sess.run(accuracy,
                                              feed_dict={x: mnist.test.images, y_: mnist.test.labels})  # 计算准确率
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("./data/", one_hot=True)  # 读入数据集
    test(mnist)  # 调用test函数


if __name__ == '__main__':
    main()
