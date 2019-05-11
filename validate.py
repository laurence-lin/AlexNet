import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import AlexNet as alexnet
import os

# Predict output class on test data, and show the real class image
classname = ['cat', 'dog'] # cat label 0, dog label 1
test_path = ['/test_data/cat_test',
             '/test_data/dog_test']  # test data floyd dataset name: test_data


test_img_path = []
test_label = []
for path in test_path:
    images = os.listdir(path)
    for img in images:
        test_img_path.append(os.path.join(path, img))
        if 'cat' in path:
            test_label.append(0)
        elif 'dog' in path:
            test_label.append(1)

# Permutation: shuffling
permutation = np.random.permutation(len(test_label))
path = []
label = []
for i in permutation:
    path.append(test_img_path[i])
    label.append(test_label[i])

test_path = path
test_label = label
test_data = []
for path in test_path:
    img = tf.read(path) # read img content
    decode = tf.image.decode_jpg(img)
    resize_img = tf.image.resize_images(decode, [227, 227])
    test_data.append(resize_img)

output_y = alexnet.out
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './tmp/model.ckpt') # restore all variables from 
    output = sess.run(output_y, feed_dict = {alexnet.input_img: test_data})
    predict = tf.argmax(output, 1)
    
    for sample in len(test_label):
        output_class = classname[predict[sample]]
        
        plt.figure(sample)
        plt.imshow(test_data[sample].eval())
        plt.title(output_class)
        
    plt.show()  











