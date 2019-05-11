import os
import numpy as np
import tensorflow as tf
import AlexNet as Alexnet
from datagenerator import ImageDataGenerator
from datetime import datetime
import glob
import matplotlib.pyplot as plt

def main():

    # Define hyperparameters
    learning_rate = 0.0007
    num_epochs =15
    train_batch_size = 200
    #dropout_rate = 0.5
    num_classes = 2  # number of classes
    display_step = 2  # 經過多少step後，計算accuracy並顯示出來

    filewriter_path = "./tmp/tensorboard"  # 存储tensorboard文件
    model_save_path = "./save_model"
    file_name_of_class = ['cat',
                          'dog']  # cat对应标签0,dog对应标签1。默認圖片包含獨立類別，可用此來建立label 列表

    train_dataset_paths = ['/my_data/cat/*.jpg',
                           '/my_data/dog/*.jpg']  # train dataset path directory

    cat_path = '/test_data/cat_test'
    dog_path = '/test_data/dog_test' 

    # create a lists for all image path
    train_image_paths = []
    train_labels = []
    # Read in all data, create list of all images
    for train_dataset in train_dataset_paths:
        img_train = glob.glob(train_dataset)  # read all files that match JPG in the directory
        train_image_paths.extend(img_train)   # Cat & Dog image training data list

    for image_path in train_image_paths:  # Create training data labels
        image_file_name = image_path.split('/')[-1]  # cut the path name and preserve final short path name, which contains discription of image
        for i in range(num_classes):
            if file_name_of_class[i] in image_file_name:  # if class name contains in the image path name (contains the image description)
                train_labels.append(i)
                break

    # create test set
    test_img_path = []
    test_label = []
    cat_test = os.listdir(cat_path)
    dog_test = os.listdir(dog_path)
    for sample in cat_test:
        test_img_path.extend(os.path.join(cat_path, sample))
        test_label.append(0)
    for sample in dog_test:
        test_img_path.extend(os.path.join(dog_path, sample))
        test_label.append(1)
    # shuffle
    permutation = np.random.permutation(len(test_img_path))
    path = []
    label = []
    for i in permutation:
        path.append(test_img_path[i])
        label.append(test_label[i])
    test_path = path
    test_label = label
    
    # Create the Dataset object that imports training data
    train_data = ImageDataGenerator(
        images=train_image_paths,
        labels=train_labels,
        batch_size=train_batch_size,
        num_classes=num_classes,
        img_format='jpg',
        shuffle=True)

    test_data = ImageDataGenerator(
            images = test_path,
            labels = test_label,
            batch_size = len(test_path)
            num_classes = num_classes,
            img_format = 'jpg',
            shuffle = True)
    
    # get Iterators
    '''
    Iterator: 設定如何run dataset的迭代
    含有兩個method:
    iterator.initialize: 將迭代重新初始化
    iterator.get_next(): 迭代到下一個sample
    '''
    # Iterator define the method we pick up element(batches of sample) from the dataset
    # Create a initializable iterator, that we could feed x ltater
    train_iterator = train_data.data.make_initializable_iterator()
    # further defined
    training_initalizer = train_iterator.initializer
    # Define next batch (element) for Iterator to get
    train_next_batch = train_iterator.get_next()

    # test iterator
    test_iterator = test_data.data.make_inializable_iterator()
    test_init = test_iterator.initializer
    test_batch = test_iterator.get_next()
    
    
    y = tf.placeholder(tf.float32, [None, num_classes])

    # alexnet
    fc8 = Alexnet.out

    # Using name scope to name the variables, so we can call these variable to show in the graph
    # loss
    with tf.name_scope('loss'):
      loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8,
                                                                            labels=y))
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
    # accuracy
    correct_pred = tf.equal(tf.argmax(fc8, 1), tf.argmax(y, 1)) # tf.argmax return the max probability(class) index of output
    # tf.equal return True, False table for the dataset
    # tf.cast convert the True, False into 1, 0 for dataset: [0, 0, 1, 0, 1, .....]
    # tf.reduce_mean compute mean value of dataset, correct prediction is 1, thus mean value is prediction accuracy of the dataset
    with tf.name_scope('accuracy'):
       accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        # reduce_mean(tensor, axis, keep_dim): compute mean value of tensor along given axis. If no axis is given, compute mean of all values in tensor.

    # Tensorboard: to show network module and training process, save module and data as ''Event" , and output to port to connect to tensorboard
    # To observe loss and accuracy along training epoch, save two scalar for summary to graph
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()  # merge all summary in graph
    writer = tf.summary.FileWriter(filewriter_path) # write all summary into hardware, create an event file in path to store the summary

    # Define number of batches
    train_batches_per_epoch = int(np.floor(train_data.size / train_batch_size))
    #test_batches_per_epoch = int(np.floor(test_data.data_size / test_batch_size))

    saver = tf.train.Saver() # saver could save all variables in a graph
    # Start training
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer.add_graph(sess.graph)  # add new session graph to current event file

        sess.run(training_initalizer)
                
        for epoch in range(num_epochs):
            sess.run(training_initalizer)  # reset iterator, get batches of train data one by one batch
            #print(sess.run(train_next_batch[0]))
            print("{}: Epoch number: {} start".format(datetime.now(), epoch + 1)) # show the clock time of each training epoch start
            train_acc = 0
            Loss = 0
            for step in range(train_batches_per_epoch):  # for each batch of samples
                img_batch, label_batch = sess.run(train_next_batch)
                batch_loss, _ = sess.run([loss_op, optimizer], feed_dict={Alexnet.input_img: img_batch,
                                                                   y: label_batch})
                # Show trainin accuracy
                batch_acc = sess.run(accuracy, feed_dict={Alexnet.input_img: img_batch, y:label_batch})
                train_acc += batch_acc
                Loss += batch_loss 
            
            train_acc /= train_batches_per_epoch
            Loss /= train_batches_per_epoch
            print('{}: Training Accuracy: = {:.4f}'.format(datetime.now(), train_acc))
            print("{}: loss = {}".format(datetime.now(), Loss))
            summary = sess.run(merged_summary, feed_dict={Alexnet.input_img: img_batch,  # save to graph the current loss & accuracy
                                                            y: label_batch,
                                                            })
            writer.add_summary(summary, epoch)  # add  new summary to current event
            
        
            # Testing accuracy
            sess.run(test_init)
            test_batch, label_batch = sess.run(test_batch)
            test_acc = sess.run(accuracy, feed_dict = {Alexnet.input_img: test_batch,
                                                       y: label_batch})
            print('Testing acc: {}'.format(test_acc))
        
        # after finished training, save the model
        save_path = saver.save(sess, './tmp/model.ckpt')
        print('Model saved in: %s'%save_path)
            


if __name__ == '__main__':
    main()

