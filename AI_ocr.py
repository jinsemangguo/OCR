# coding=utf-8
import tensorflow as tf
import os
import pandas as pd
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class VerificationCode(object):
    def __init__(self):
        pass

    def parse_csv(self):
        code_data = pd.read_csv('./GenPics/labels.csv',names=['file_num','char_code'],index_col=['file_num'])

        vc_code = code_data['char_code']

        letter_list = []
        for line in vc_code:
            line = line.strip('\n').strip()
            letters = []
            for c in line:
                diff = ord(c) - ord('A')
                letters.append(diff)

            letter_list.append(letters)

        # print(letter_list)
        code_data['verification_code'] = letter_list
        # print(code_data)
        return code_data

    def picture_read(self):
        filenames = os.listdir('./GenPics/')
        file_list = ['./GenPics/' + name for name in filenames if name[-3:] == 'jpg']

        file_queue = tf.train.string_input_producer(file_list)

        reader = tf.WholeFileReader()
        key,value = reader.read(file_queue)

        image = tf.image.decode_jpeg(value)
        image.set_shape([20,80,3])

        filename_batch,image_batch = tf.train.batch([key,image],batch_size=40,num_threads=2,capacity=70)
        return filename_batch,image_batch

    def file_to_label(self,filename,label_data):
        labels = []
        for name in filename:
            index,_ = os.path.splitext(os.path.basename(name))
            code = label_data.loc[int(index),'verification_code']
            labels.append(code)

        return np.array(labels)

    def init_weights(self,shape):
        return tf.Variable(initial_value=tf.random_normal(shape=shape,mean=0.0,stddev=0.1))

    ''' Building a network'''

    def cnn_model(self,x):
        with tf.variable_scope('conv_1'):
            conv1_w = self.init_weights([5,5,3,32])
            conv1_b = self.init_weights([32])

            x_conv1 = tf.nn.conv2d(x,conv1_w,strides=[1,1,1,1],padding='SAME') + conv1_b

            # relu
            x_relu1 = tf.nn.relu(x_conv1)

            x_pool1 = tf.nn.max_pool(x_relu1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        with tf.variable_scope('conv_2'):
            conv2_w = self.init_weights([5,5,32,64])
            conv2_b = self.init_weights([64])

            x_conv2 = tf.nn.conv2d(x_pool1,conv2_w,strides=[1,1,1,1],padding='SAME') + conv2_b

            x_relu2 = tf.nn.relu(x_conv2)

            x_pool2 = tf.nn.max_pool(x_relu2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

        # [None, 5, 20, 64]
        with tf.variable_scope('full_connection'):
            x_fc = tf.reshape(x_pool2,[-1,5 * 20 * 64])

            fc_w = self.init_weights([5 * 20 * 64,26 * 4])
            fc_b = self.init_weights([26 * 4])

            y_predict = tf.matmul(x_fc,fc_w) + fc_b

        return y_predict

    # Test model
    def img_test_demo(self):
        filenames = os.listdir('./GenPics')
        file_list = ['./GenPics/' + name for name in filenames if name[-3:] == 'jpg']
        file_queue = tf.train.string_input_producer(file_list)

        reader = tf.WholeFileReader()
        key,value = reader.read(file_queue)

        image = tf.image.decode_jpeg(value)
        image.set_shape([20,80,3])

        # Use a single test, so only one picture is passed in one batch,batch_size=1
        file_batch,image_batch = tf.train.batch([key,image],batch_size=1,num_threads=1,capacity=32)

        # Define network
        with tf.variable_scope('original_data'):
            x = tf.placeholder(dtype=tf.float32,shape=[None,20,80,3])
            y_true = tf.placeholder(dtype=tf.float32,shape=[None,26 * 4])
            y_predict = self.cnn_model(x)

        with tf.variable_scope('loss'):
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_predict))

        with tf.variable_scope('optimizer'):
            train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

        with tf.variable_scope('accuracy'):
            equal_list = tf.reduce_all(
                tf.equal(tf.argmax(tf.reshape(y_true,[-1,4,26]),axis=2),
                         tf.argmax(tf.reshape(y_predict,[-1,4,26]),axis=2)),
                axis=1
            )
            accuracy = tf.reduce_mean(tf.cast(equal_list,tf.float32))

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)

            label_data = self.parse_csv()

            if os.path.exists('./models/captcha/checkpoint'):
                saver.restore(sess,'./models/captcha/captcha')

            coo = tf.train.Coordinator()

            threads = tf.train.start_queue_runners(sess=sess,coord=coo)

            for epoch in range(100):
                filename,image = sess.run([file_batch,image_batch])
                label = self.file_to_label(filename,label_data)

                label_onehot = tf.reshape(tf.one_hot(label,26),[-1,4 * 26]).eval()

                y_1 = tf.argmax(tf.reshape(sess.run(y_true,feed_dict={x: image,y_true: label_onehot}),[-1,4,26]),
                                axis=2).eval()[0]
                y_2 = tf.argmax(tf.reshape(sess.run(y_predict,feed_dict={x: image,y_true: label_onehot}),[-1,4,26]),
                                axis=2).eval()[0]

                # Convert a list of numbers of real values to real letters
                letter1 = []
                for i in y_1:
                    char_i = chr(i + 65)
                    letter1.append(char_i)
                label1 = ''.join(letter1)

                # Convert a list of numbers of predicted values to letters
                letter2 = []
                for i in y_2:
                    char_i = chr(i + 65)
                    letter2.append(char_i)
                label2 = ''.join(letter2)

                # Print the two strings and compare them.
                print('%d image test sample, True value = %s, Forecast value = %s' % (epoch + 1,label1,label2))

            coo.request_stop()
            coo.join(threads)

    def md_run_demo(self):
        self.img_test_demo()


if __name__ == '__main__':
    vc = VerificationCode()
    vc.md_run_demo()
