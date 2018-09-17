import tensorflow as tf
tf.set_random_seed(777)  # for reproducibility

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# Try to find values for W and b to compute y_data = W * x_data + b
# We know that W should be 1 and b should be 0
# But let's use TensorFlow to figure it out
W = tf.Variable(tf.random_normal([1]), name='weight')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Our hypothesis for linear model X * W
hypothesis = X * W

# cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
# W -= W - a * 1/m sigma(wx - y)x
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

'''
# 위 부분을 매번 미분할 필요 없이 아래와 같이 간략히 쓸수 있음.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
update = optimizer.minimize(cost)
'''

'''
# tensorflow가 주는 gradient에서, 값을 변경하고 싶을때,
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
gvs = optimizer.compute_gradients(cost)   # get gradients
# 수정하고 싶다면 여기서 수정
update = optimizer.apply_gradients(gvs)  #apply gradients
'''

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(21):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))
