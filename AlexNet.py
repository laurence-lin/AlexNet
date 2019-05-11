import tensorflow as tf
import numpy as np

# Alexnet for cifar dataset
num_class = 2
input_img = tf.placeholder(tf.float32, [None, 227, 227, 3])
#dropout = tf.placeholder(tf.float32)

c1_w = tf.Variable(tf.truncated_normal(shape = [11, 11, 3, 96], stddev = 0.1), name = 'c1_w')
c1_b = tf.Variable(tf.truncated_normal([96]))
c1_out = tf.nn.conv2d(input_img, c1_w, strides = [1, 4, 4, 1], padding= 'SAME') + c1_b
c1_out = tf.nn.relu(c1_out)
    # output size: batch*
    
lrn1 = tf.nn.lrn(c1_out, 
                 alpha = 1e-4,
                 beta = 0.75,
                 depth_radius = 2,
                 bias = 2.0)

    # Max pooling layer 1
p1 = tf.nn.max_pool(lrn1,
                    ksize = [1,3,3,1],
                    strides = [1,2,2,1],
                    padding = 'SAME'
                    )
    # batch*16*16*96
    # padding
    # We don't have two GPUs, so no need to split to half
    # Convolution layer 2
c2_w = tf.Variable(tf.truncated_normal([5, 5, 96, 256]))
c2_b = tf.Variable(tf.truncated_normal([256]))
c2_out = tf.nn.conv2d(p1, c2_w, [1,1,1,1], 'SAME') + c2_b
c2_out = tf.nn.relu(c2_out)
    # output size: batch*16*16*256
lrn2 = tf.nn.lrn(c2_out, 
                 alpha = 1e-4,
                 beta = 0.75,
                 depth_radius = 2,
                 bias = 2.0)

    # Max pooling layer 2
p2 = tf.nn.max_pool(lrn2,
                    ksize=[1, 3, 3, 1],
                    strides=[1, 2, 2, 1],
                    padding='SAME'
                    )
    # output size: 
    # Convolution layer 3: In layer 3, no need to split 2 sections, full connected layer
c3_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 384]))
c3_b = tf.Variable(tf.truncated_normal([384]))
c3_out = tf.nn.conv2d(p2, c3_w, strides=[1, 1, 1, 1], padding='SAME') + c3_b
c3_out = tf.nn.relu(c3_out)
    # 10000*8*8*384

    # Convolution layer 5: final convolution layer, with a max pooling layer
c4_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 384]))
c4_b = tf.Variable(tf.truncated_normal([384]))
c4_out = tf.nn.conv2d(c3_out, c4_w, strides=[1, 1, 1, 1], padding='SAME') + c4_b
c4_out = tf.nn.relu(c4_out)
    
c5_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 384,256]))
c5_b = tf.Variable(tf.truncated_normal([256]))
c5_out = tf.nn.conv2d(c4_out, c5_w, strides=[1, 1, 1, 1], padding='SAME') + c5_b
c5_out = tf.nn.relu(c5_out)
    # 10000*8*8*256
    
    # flatten output to fully connected layer
flatten5 = tf.layers.flatten(c5_out)  # we can do flatten with this and tf.reshape, but layer.flatten don't need batch size
features = 4096
# fully connected layer 6: with dropout layer
w6 = tf.Variable(tf.truncated_normal(shape = [57600, features])) # initial parameters cannot be placeholder, should be fixed value
b6 = tf.Variable(tf.truncated_normal([features]))
f6_out = tf.matmul(flatten5, w6) + b6
f6_out = tf.nn.relu(f6_out)
#f6_out = tf.nn.dropout(f6_out, dropout)

    # fully connected layer 7: followed by dropout layer
w7 = tf.Variable(tf.truncated_normal(shape=[features, features]))
b7 = tf.Variable(tf.truncated_normal([features]))
f7_out = tf.matmul(f6_out, w7) + b7
f7_out = tf.nn.relu(f7_out)
#f7_out = tf.nn.dropout(f7_out, dropout)

    # output layer
w_out = tf.Variable(tf.truncated_normal(shape = [features, num_class]))
b_out = tf.Variable(tf.truncated_normal([num_class]))
out = tf.matmul(f7_out, w_out) + b_out



















