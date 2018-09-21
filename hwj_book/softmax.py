import tensorflow as tf
# Import MINST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

# input 28*28 = 784 per image
# training set images 55000 * 784
# training set labels 55000 * 10

# hyper parameters
learning_rate = 0.5
batch_size = 100
training_epochs = 1000

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, [None, 784])

# ground truth
y = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))  # weights
b = tf.Variable(tf.zeros([10]))       # bias

# hypothesis
pred = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

tf.global_variables_initializer().run()

for epoch in range(training_epochs):
  
  # total_batch = int(mnist.train.num_examples/batch_size)
  # apply batch gradient descent
  # for i in range(total_batch):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  optimizer.run({x: batch_xs, y: batch_ys})

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

print '==--==--=='

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
