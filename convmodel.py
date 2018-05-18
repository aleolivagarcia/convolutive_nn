# -*- coding: utf-8 -*-

# Sample code to use string producer.

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """

    o_h = np.zeros(n)
    o_h[x] = 1.
    return o_h



num_classes = 3
batch_size = 4
batch_test_size = 42
lrate=0.01


# --------------------------------------------------
#
#       DATA SOURCE
#
# --------------------------------------------------

def dataSource(paths, batch_size):
    min_after_dequeue = 10
    capacity = min_after_dequeue + 3 * batch_size

    example_batch_list = []
    label_batch_list = []

    for i, p in enumerate(paths):
        filename = tf.train.match_filenames_once(p)
        filename_queue = tf.train.string_input_producer(filename, shuffle=False)
        reader = tf.WholeFileReader()
        _, file_image = reader.read(filename_queue)
        image, label = tf.image.decode_jpeg(file_image), one_hot([i], num_classes) # [float(i)]
        #image = tf.image.resize_image_with_crop_or_pad(image, 80, 140)
        image = tf.reshape(image, [80, 140, 1])
        image = tf.to_float(image) / 255. - 0.5
        example_batch, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size, capacity=capacity,
                                                            min_after_dequeue=min_after_dequeue)

        example_batch_list.append(example_batch)
        label_batch_list.append(label_batch)


    example_batch = tf.concat(values=example_batch_list, axis=0)
    label_batch = tf.concat(values=label_batch_list, axis=0)

    return example_batch, label_batch


# --------------------------------------------------
#
#       MODEL
#
# --------------------------------------------------

def myModel(X, batch, reuse=False):
    with tf.variable_scope('ConvNet', reuse=reuse):
        o1 = tf.layers.conv2d(inputs=X, filters=32, kernel_size=3, activation=tf.nn.relu)
        o2 = tf.layers.max_pooling2d(inputs=o1, pool_size=2, strides=2)
        o3 = tf.layers.conv2d(inputs=o2, filters=64, kernel_size=3, activation=tf.nn.relu)
        o4 = tf.layers.max_pooling2d(inputs=o3, pool_size=2, strides=2)

        h = tf.layers.dense(inputs=tf.reshape(o4, [batch * 3, 18 * 33 * 64]), units=50, activation=tf.nn.relu)
        y = tf.layers.dense(inputs=h, units=3, activation=tf.nn.softmax)
    return y


example_batch_train, label_batch_train = dataSource(["Dataset/Train/0/*.jpg", "Dataset/Train/1/*.jpg", "Dataset/Train/2/*.jpg"], batch_size=batch_size)
example_batch_valid, label_batch_valid = dataSource(["Dataset/Validation/0/*.jpg", "Dataset/Validation/1/*.jpg", "Dataset/Validation/2/*.jpg"], batch_size=batch_size)
example_batch_test, label_batch_test = dataSource(["Dataset/Test/0/*.jpg", "Dataset/Test/1/*.jpg", "Dataset/Test/2/*.jpg"], batch_size=batch_test_size)

example_batch_train_predicted = myModel(example_batch_train, batch_size, reuse=False)
example_batch_valid_predicted = myModel(example_batch_valid, batch_size, reuse=True)
example_batch_test_predicted = myModel(example_batch_test, batch_test_size, reuse=True)


label_batch_train = tf.cast(label_batch_train, tf.float32)
label_batch_valid = tf.cast(label_batch_valid, tf.float32)
label_batch_test = tf.cast(label_batch_test, tf.float32)

cost = tf.reduce_sum(tf.square(example_batch_train_predicted - label_batch_train))
cost_valid = tf.reduce_sum(tf.square(example_batch_valid_predicted - label_batch_valid))
# cost = tf.reduce_mean(-tf.reduce_sum(label_batch * tf.log(y), reduction_indices=[1]))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrate).minimize(cost)

# --------------------------------------------------
#
#       TRAINING
#
# --------------------------------------------------

# Add ops to save and restore all the variables.

saver = tf.train.Saver()

error=7
errorArray = []
errorArray.append(error)
epochs = 1
maxIter=300

with tf.Session() as sess:
    print("Parámetros:\nCapas convolutivas: 2\nCapas ocultas: 1\n"
          "Número de neuronas en la capa: 50\nLearning Rate:", lrate)

    file_writer = tf.summary.FileWriter('./logs', sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    while True:
        sess.run(optimizer)


        errorPrev = error
        error = sess.run(cost_valid)
        errorArray.append(error)
        '''
        print("Iteración:", epochs)
        print("Error =", error)
        print(abs(error-errorPrev)/errorPrev)
        '''
        if (abs(error-errorPrev)/errorPrev) < 0.01 or epochs >= maxIter:
                break

        epochs += 1

    test_result = sess.run(example_batch_test_predicted)

    aciertos = 0
    for label, nn in zip(label_batch_test.eval(), test_result):
        if np.argmax(nn) == np.argmax(label):
            aciertos += 1

    precision = aciertos / len(label_batch_test.eval()) * 100
    print("Precisión de la red:", precision, "%")


    save_path = saver.save(sess, "./tmp/model.ckpt")
    #print("Model saved in file: %s" % save_path)

    coord.request_stop()
    coord.join(threads)



plt.subplot(1, 2, 1)
plt.plot(errorArray)
plt.xlabel("Epochs")
plt.ylabel("Error de validación")
plt.title("Variación del error")
plt.show()
