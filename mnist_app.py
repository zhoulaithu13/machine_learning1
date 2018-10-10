# 实现手写数字识别

import tensorflow as tf
import numpy as np
from PIL import Image
import mnist_backward
import mnist_forward


def restore_model(testPicArr):
    with tf.Graph().as_default() as g:  # 重现计算图
        x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
        y = mnist_forward.forward(x, None)  # 计算输出y
        preValue = tf.argmax(y, 1)  # 最大值即为预测结果

        variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)  # 实例化带有滑动平均值的saver

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})  # 待识别图片喂入网络
                return preValue
            else:
                print("No checkpoint file found")
                return -1  # 加载ckpt


def pre_pic(picName):
    img = Image.open(picName)  # 打开原始图片
    reIm = img.resize((28, 28), Image.ANTIALIAS)  # 用消除锯齿的方法resize
    im_arr = np.array(reIm.convert('L'))  # 变成灰度图，再转换为矩阵
    threshold = 50  # 阈值
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]  # 反差色，模型为黑底白字，输入为白底黑字
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255  # 仅有纯白色或纯黑色点，滤掉噪声

    nm_arr = im_arr.reshape([1, 784])  # 整理形状为1行784列
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)  # 像素要求为0、1之间的浮点数

    return img_ready


def application():
    testNum = int(input("input the number of test pictures:"))  # 输入要识别的图片数

    for i in range(testNum):
        testPic = input("the path of test picture:")  # 识别图片的路径
        testPicArr = pre_pic(testPic)  # 图片预处理
        preValue = restore_model(testPicArr)  # 喂入神经网络
        print("The prediction number is", preValue)


def main():
    application()


if __name__ == '__main__':
    main()
