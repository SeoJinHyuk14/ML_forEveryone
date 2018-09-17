import tensorflow as tf
import random
import matplotlib.pyplot as plt
tf.set_random_seed(777)  # for reproducibility

from tensorflow.examples.tutorials.mnist import input_data
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10

# MNIST data image of shape 28 * 28 = 784
X = tf.placeholder(tf.float32, [None, 784])
# 0 - 9 digits recognition = 10 classes
Y = tf.placeholder(tf.float32, [None, nb_classes])

'''
# using softmax
W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

# Hypothesis (using softmax)
hypothesis = tf.nn.softmax(tf.matmul(X, W) + b)

# cross entropy H(y) = - sum( Y log(y) )
# tf.reduce_mean으로 batch의 모든 예시에 대한 평균 계산
cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
'''

'''
# using softmax shortly and using AdamOptimizer
W = tf.Variable(tf.random_normal([784, 10]))
b = tf.Variable(tf.random_normal([10]))

hypothesis = tf.matmul(X, W) + b

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
'''

# '''
# using NN
# xavier : http://stackoverflow.com/questions/33640581/how-to-do-xavier-initialization-on-tensorflow
output_cnt = 256 # 히든 레이어에서 출력 값은 하고 싶은대로 설계. 다만 다음 레이어의 인풋과 값이 같아야함.
output_cnt2 = 256
# W1 = tf.Variable(tf.random_normal([784, output_cnt]))
W1 = tf.get_variable("W1", shape=[784, output_cnt], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([output_cnt]))
L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

# W2 = tf.Variable(tf.random_normal([output_cnt, output_cnt2]))
W2 = tf.get_variable("W2", shape=[output_cnt, output_cnt2], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([output_cnt2]))
L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

# W3 = tf.Variable(tf.random_normal([output_cnt2, 10]))
W3 = tf.get_variable("W3", shape=[output_cnt2, 10], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b3

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# '''

# Test model
# 특정 축에 따라 가장 큰 값이 학습한 데이터와 실제 데이터와 같은지 boolean return
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
# Calculate accuracy
# true => 1, false => 0로 cast해서, 평균값 계산하여 0~1 사이값으로 정확성
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# parameters
batch_size = 100    # 메모리 부담 안가게 100개씩 불러와서 학습
training_epochs = 15
total_batch = int(mnist.train.num_examples / batch_size)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print("Learning finished")

    # Test the model using test sets
    print("Accuracy: ", accuracy.eval(session=sess, feed_dict={
        X: mnist.test.images, Y: mnist.test.labels}))

    # Get one and predict
    r = random.randint(0, mnist.test.num_examples - 1)
    print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
    print("Prediction: ", sess.run(
        tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))

    for index, pixel in enumerate(mnist.test.images[r]):
        if index % 28 == 0:
            print('\n')
        else:
            print("%.2f" % pixel, end=" ")
    print('\n')

    plt.imshow(
        mnist.test.images[r:r + 1].reshape(28, 28),
        cmap='Greys',
        interpolation='nearest')
    plt.show()
