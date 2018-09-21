import cifar10, cifar10_input
import tensorflow as tf 
import numpy as np
import time 

# hyper parameters
max_steps = 3000
hatch_size = 128
data_dir = '/tmp/cifar10_data/cifar-10-batches-bin'