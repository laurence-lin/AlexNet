import tensorflow as tf
import numpy as np
import AlexNet as alexnet
import os
import matplotlib 
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Predict output class on test data, and show the real class image
classname = ['cat', 'dog'] # cat label 0, dog label 1
test_path = ['/test_data/cat_test',
             '/test_data/dog_test']  # test data floyd dataset name: test_data

model_path = '/model_path/tmp/'
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
test_label = tf.convert_to_tensor(np.array(label))
test_data = []
for path in test_path:
    img = tf.read_file(path) # read img content
    decode = tf.image.decode_jpeg(img, channels = 3)
    resize_img = tf.image.resize_images(decode, [227, 227])
    test_data.append(resize_img.eval(session = tf.Session())) 

test_data = np.array(test_data)

output_y = alexnet.out
saver = tf.train.Saver() # model saver could also restore pretrained models
correct_pred = tf.equal(tf.argmax(output_y, 1), tf.argmax(test_label, 1))
with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
       
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('/model_path/tmp/test_model.ckptyt.meta')  # load network architecture
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, tf.train.latest_checkpoint(model_path)) # load variables, gradients
    output, accuracy = sess.run([output_y, accuracy], feed_dict = {alexnet.input_img: test_data, 
                                })
    
    print('Testing accuracy:', accuracy)
    predict = tf.argmax(output, 1)
    predict = predict.eval()
    # show test image result
    test_data = test_data.astype(int) # show image by integer pixel value
    for sample in range(len(test_label)):
        output_class = classname[predict[sample]]
        
        save_name = str(sample) + '.jpg'
        plt.figure(sample)
        plt.imshow(test_data[sample, :, :, :])
        plt.title(output_class)
        plt.savefig(save_name)
        
      











