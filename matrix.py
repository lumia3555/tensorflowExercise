import tensorflow as tf
import numpy
# hello = tf.constant('hello tensorflow')
# sess = tf.Session()
# print sess.run(hello)

a = tf.constant(2)
b = tf.constant(3)

m1 = tf.constant([[3,4]])
m2 = tf.constant([[5],[6]])

xx = numpy.asarray([1,2,3,4,5])

l1 = tf.constant([1,2,3,4])
product = tf.matmul(m2, m1)

with tf.Session() as sess:
  print "a=2 b=3"
  result = sess.run(product)
  print result
  print xx
