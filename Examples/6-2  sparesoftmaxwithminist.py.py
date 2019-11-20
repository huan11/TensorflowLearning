# -*- coding: utf-8 -*-
"""
2019 11 20
@author: MH
代码用到的东西： MNIST 数据集
              pylab
              tensorflow

"""
#tensorflow中包含了MNIST数据集的下载代码
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/")

print ('输入数据:',mnist.train.images)
print ('输入数据打shape:',mnist.train.images.shape)

#pylab 是类似于 matlab 这样一个综合性的平台， pylab 把 Python, NumPy, SciPy, Matplotlib都集合起来了
import pylab 
im = mnist.train.images[1]
im = im.reshape(-1,28)
pylab.imshow(im)
pylab.show()


print ('输入数据打shape:',mnist.test.images.shape)
print ('输入数据打shape:',mnist.validation.images.shape)


import tensorflow as tf #导入tensorflow库
tf.reset_default_graph()
########################################################################## 构建模型
#################################################### 1 输入节点
# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data维度 28*28=784
y = tf.placeholder(tf.int32, [None]) # 0-9 数字=> 10 classes
#################################################### 2 模型参数
# Set model weights
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.zeros([10]))
#################################################### 3 前向结构
z= tf.matmul(x, W) + b   # 定义运算
pred = tf.nn.softmax(z) # Softmax分类
##################################################### 4 反向优化
# Minimize error using cross entropy
#cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=z))
#参数设置
learning_rate = 0.01  #定义一个学习率，代表调整参数的速度
# 使用梯度下降优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

training_epochs = 25  #设置迭代次数
batch_size = 100
display_step = 1

########################################################################## 训练模型
# 启动session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())# Initializing OP

    # 启动循环开始训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # 遍历全部数据集
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # 显示训练中的详细信息
        if (epoch+1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

    print( " Finished!")
















