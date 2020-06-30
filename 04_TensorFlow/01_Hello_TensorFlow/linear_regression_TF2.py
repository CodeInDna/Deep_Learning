import tensorflow as tf
import numpy as np
tf.compat.v1.disable_eager_execution()
print('\nHello! This script is being run by a kernel running TensorFlow version {}!\n'.format(tf.__version__))

np.random.seed(1)

LEARNING_RATE = 0.01
EPOCHS        = 100

x_train = np.array(range(-5, 6))
y_train = np.array(range(-10, 12, 2)) + np.random.randn(x_train.shape[0])
n = x_train.shape[0]

x = tf.compat.v1.placeholder('float')
y = tf.compat.v1.placeholder('float')
w = tf.Variable(np.random.randn(), name='weight')
b = tf.Variable(np.random.randn(), name='bias')
y_hat = tf.add(tf.multiply(x, w), b)

mse = tf.reduce_sum(input_tensor=tf.pow(y_hat - y, 2))/n
optimizer = tf.compat.v1.train.GradientDescentOptimizer(LEARNING_RATE).minimize(mse)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    for epoch in range(EPOCHS):
        for (x_value, y_value) in zip(x_train, y_train):
            sess.run(optimizer, feed_dict={x: x_value, y: y_value})

    final_mse = sess.run(mse, feed_dict={x: x_train, y: y_train})
    print('Regression complete.\nValues at convergence:\nMSE: {0:6.4}\nw: {1:8.4}\nb: {2:10.4}\n'.format(final_mse, sess.run(w), sess.run(b)))
print('Script ran successfully. Goodbye!\n')
