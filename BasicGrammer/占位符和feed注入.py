"""
import tensorflow as tf
a = tf.constant(3)
b = tf.constant(4)
# 使用 With 语法开启 Session
with tf.Session() as sess:
    print("相加：%i" % sess.run(a+b))
    print("相乘：%i" % sess.run(a*b))
"""    
    
    
    
# 演示注入机制  使用注入机制， 将具体的实参注入到相应的 placeholder 中      
import tensorflow as tf
# 通过tf.placeholder 为这些操纵创建了占位符  
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

add = tf.add(a,b)
mul = tf.multiply(a,b)
# 使用 feed 机制 将具体的数值（3 和 4 ） 通过占位符传入 ，并进行相关的运算
with tf.Session() as sess:
    # 通过feed机制   把具体的值放到占位符里
    print("相加：%i" % sess.run(add,feed_dict={a:3,b:4}))
    print("相乘：%i" % sess.run(mul,feed_dict={a:3,b:4}))