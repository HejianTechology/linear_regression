import tensorflow as tf
import data


# 获取数据
x_data,y_data,x_test_data,y_test_data = data.get_dataset()
data_size = len(x_data)

# 开始建立模型
wight = tf.Variable(tf.random_uniform([1]),tf.float32)
bias = tf.Variable(tf.zeros([1]),tf.float32)
y_ = tf.multiply(wight,x_data) + bias

# 计算loss,并进行梯度下降
loss = tf.reduce_mean(tf.square(y_data-y_))
train = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 计算误差和准确率
y_predict = wight*x_test_data + bias
error = tf.reduce_mean(tf.abs(y_test_data - y_predict))
accuracy = 1-error

with tf.Session() as sess:
    # 初始化所有变量
    sess.run(tf.global_variables_initializer())

    # 开始进行训练
    for train_step in range(1000):
        sess.run(train)
        if train_step%50 ==0:
            print(sess.run(wight),sess.run(bias))

    # 打印误差
    print(sess.run(error),sess.run(accuracy))

