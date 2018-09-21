from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()

# hyper parameters
learning_rate = 0.3
batch_size = 100
training_epochs = 3000

in_units = 784
h1_units = 300

# manually set weights/bias for hidden layer
# map from input to h1
W1 = tf.Variable(tf.truncated_normal([in_units, h1_units], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1_units]))
# map from h1 to output
W2 = tf.Variable(tf.zeros([h1_units, 10]))
b2 = tf.Variable(tf.zeros([10]))

x = tf.placeholder(tf.float32, [None, in_units])
# set dropout rate
keep_prob = tf.placeholder(tf.float32)

# h1 model
hidden1 = tf.nn.relu(tf.matmul(x, W1) + b1)
hidden1_drop = tf.nn.dropout(hidden1, keep_prob)

pred = tf.nn.softmax(tf.matmul(hidden1_drop, W2) + b2)

# ground truth
y = tf.placeholder(tf.float32, [None, 10])

# cost function
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=[1]))

# use Adagrad instead of GradientDescent
optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy)

tf.global_variables_initializer().run()

# train
for epoch in range(training_epochs):
  
  #total_batch = int(mnist.train.num_examples/batch_size)

  #for i in range(total_batch):
  batch_xs, batch_ys = mnist.train.next_batch(batch_size)
  optimizer.run({x: batch_xs, y: batch_ys, keep_prob: 0.75})

# predict
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval({x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0}))
