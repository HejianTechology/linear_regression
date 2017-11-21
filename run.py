import tensorflow as tf
import data


# 实现你的代码
W = tf.Variable(tf.random_uniform([1]),dtype=tf.float32)
b = tf.Variable(tf.random_uniform([1]),dtype=tf.float32)
x = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32)
y = W * x + b

cost = tf.reduce_sum(tf.pow((y_-y),2))

train = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(1000):
        x_train, y_train, x_test, y_test = data.get_dataset()
        sess.run(train, feed_dict={x: x_train, y_: y_train})

        accuracy = 1-tf.reduce_mean(tf.abs(y-y_))
        print(sess.run(accuracy, feed_dict={x: x_test, y_: y_test}))
        print(sess.run(W),sess.run(b))