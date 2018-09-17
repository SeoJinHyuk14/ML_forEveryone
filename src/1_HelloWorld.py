import tensorflow as tf

hello = tf.constant("Hello, Tensorflow")    #기본 그래프에서, Hello, Tensorflow라는 노드를 생성한거임.

sess = tf.Session() #seart a TF session

print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
node3 = tf.add(node1, node2)

# 텐서가 출력 됨
print("node1:" , node1, "node2:", node2)
print("node3:", node3)

# 연산은 세션으로 해야함.
sess = tf.Session()
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

# 변수로 할당해서 실행
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1,3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, feed_dict={a: 3, b:4.5}))