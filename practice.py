import numpy as np
import tensorflow as tf
a = []
a.append(1)
print(a)
a.extend([2])
print(a)
path = '/img/cat/test'

x = tf.truncated_normal([3,2])
with tf.Session() as sess:
    print(x.eval())