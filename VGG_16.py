import numpy as np
import matplotlib as plt
import tensorflow as tf

class VGG_16:
    def __init__(self,imgs):
        self.images = imgs
        self.convlayers()

    def convlayers(self):
        self.parameters = []
        with tf.name_scope('pre_process')as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1,1,1,3], name = 'imgs_mean')
            images = self.images - mean
        # normal data (các bức ảnh sẽ được trừ đi giá trị trung bình)

        with tf.name_scope('conv1_1')as scope:
            # tf.truncate_normal: khởi tạo giá trị cho filter
            kernel = tf.Variable(tf.truncated_normal([3,3,3,64],
                                                     stddev=1e-1,
                                                     dtype=tf.float32), name = 'weight')
            conv = tf.nn.conv2d(images,kernel,[1,1,1,1],padding='SAME') #nhân chập
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32,shape=[64]),name='biases') #cộng bias
            out_temp = tf.nn.bias_add(conv, biases)
            conv1_1 = tf.nn.relu(out_temp, name= scope)#RELU
            self.parameters += [kernel, biases]

        with tf.name_scope('conv1_2')as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,64],
                                                     stddev=1e-1,
                                                     dtype=tf.float32), name='weight')
            conv = tf.nn.conv2d(conv1_1,kernel,[1,1,1,1], padding='SAME')#strides = [1, strx, stry,1]
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[64]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv1_2 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]
        self.pool1=tf.nn.max_pool(conv1_2,
                             ksize=[1,2,2,1],
                             strides=[1,2,2,1],
                             padding='SAME',
                             name='pool1')

        with tf.name_scope('conv2_1')as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,128],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(self.pool1, kernel,[1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv2_1 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv2_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,128],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv=tf.nn.conv2d(conv2_1, kernel, [1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[128]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv2_2 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]
        self.pool2 = tf.nn.max_pool(conv2_2,
                                    ksize=[1,2,2,1],
                                    strides=[1,2,2,1],
                                    padding='SAME',
                                    name='pool2')

        with tf.name_scope('conv3_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 256],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv3_1 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv3_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,256],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(conv3_1, kernel, [1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv3_2 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv3_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,256],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(conv3_2, kernel, [1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[256]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv3_3 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]
        self.pool3 = tf.nn.max_pool(conv3_3,
                                    ksize=[1,2,2,1],
                                    strides=[1,1,1,1],
                                    padding='SAME',
                                    name='pool3')

        with tf.name_scope('conv4_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,512],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv4_1 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv4_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,512],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(conv4_1, kernel, [1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv4_2 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv4_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 512],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv4_3 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]
        self.pool4 = tf.nn.max_pool(conv4_3,
                                    ksize=[1,2,2,1],
                                    strides=[1,1,1,1],
                                    padding='SAME',
                                    name='pool4')

        with tf.name_scope('conv5_1') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,512],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv5_1 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv5_2') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,512],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(conv5_1, kernel, [1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv5_2 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]

        with tf.name_scope('conv5_3') as scope:
            kernel = tf.Variable(tf.truncated_normal([3,3,64,512],
                                                     stddev=1e-1,
                                                     dtype=tf.float32,
                                                     name='weight'))
            conv = tf.nn.conv2d(conv5_2, kernel, [1,1,1,1], padding='SAME')
            biases = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[512]), name='biases')
            out_temp = tf.nn.bias_add(conv, biases)
            conv5_3 = tf.nn.relu(out_temp, name=scope)
            self.parameters += [kernel, biases]
        self.pool5 = tf.nn.max_pool(conv5_3,
                                    ksize=[1,2,2,1],
                                    strides=[1,1,1,1],
                                    padding='SAME',
                                    name='pool5')

    def fc_layers(self):

        #fc1
        with tf.name_scope('fc1')as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fclw = tf.Variable(tf.truncated_normal([7,7,512,4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1),name='weights' )
            fclb = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1,shape])
            fcl1 = tf.nn.bias_add(tf.matmul(pool5_flat, fclw, fclb))
            self.fcl = tf.nn.relu(fcl1)
            self.parameters += [fclw, fclb]
        #fc2
        with tf.name_scope('fc2')as scope:
            fc2 = tf.Variable(tf.truncated_normal([4096,4096]))
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fclw = tf.Variable(tf.truncated_normal([7, 7, 512, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fclb = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fcl2 = tf.nn.bias_add(tf.matmul(pool5_flat, fclw, fclb))
            self.fcl = tf.nn.relu(fcl2)
            self.parameters += [fclw, fclb]


        #fc3
        with tf.name_scope('fc3')as scope:
            fc3 = tf.Variable(tf.truncated_normal([1000, 1000]))
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            fclw = tf.Variable(tf.truncated_normal([7, 7, 512, 4096],
                                                   dtype=tf.float32,
                                                   stddev=1e-1), name='weights')
            fclb = tf.Variable(tf.constant(1.0, shape=[4096], dtype=tf.float32), name='biases')
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fcl3 = tf.nn.bias_add(tf.matmul(pool5_flat, fclw, fclb))
            self.fcl = tf.nn.relu(fcl3)
            self.parameters += [fclw, fclb]


