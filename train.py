import numpy as np
import datetime, time
import tensorflow as tf
import sys, os, re, gzip, json
from sklearn.utils import shuffle


def load_json(file_name):
    with open(file_name + '.json', 'rb') as fin:
        json_bytes = fin.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
        return data

data = np.load('btcusd_dataset.npy')

test_mode = True
if test_mode:
    data = data[np.random.randint(0,len(data),5000),:]

def generate_data(window_size = 288, step_size = 1, dev_size = 0.04, test_size = 0.04):

    dev_size = round(len(data)*dev_size)
    test_size = round(len(data)*test_size)

    slices = np.array([[i,i+window_size] for i in range(0,len(data)-window_size,step_size)])
    slices = shuffle(slices)

    features_train, features_dev, features_test = np.split(slices, [-(test_size+dev_size), -test_size])

    train_fea = tf.constant(features_train, dtype = tf.float32)
    train_lab = tf.constant([window_size]*len(features_train))

    dev_fea = tf.constant(features_dev, dtype = tf.float32)
    dev_lab = tf.constant([window_size]*len(features_dev))

    test_fea = tf.constant(features_test, dtype = tf.float32)
    test_lab = tf.constant([window_size]*len(features_test))

    return ((train_fea, train_lab),(dev_fea, dev_lab),(test_fea, test_lab))


def input_parser(start_index,label_):
    slices = tf.linspace(start=tf.gather_nd(start_index,[0]), stop=tf.gather_nd(start_index,[1])-1.0, num=289-1)
    norm = tf.gather(data[:,0], tf.cast(tf.gather_nd(start_index,[0]) ,tf.int32))
    day_ = tf.gather(data[:,:-1], tf.cast(slices, tf.int32))/norm

    pred_ = tf.gather(data[:,-1:], tf.cast(tf.gather_nd(start_index,[1])-1.0, tf.int32 ))/norm
    return day_, pred_

train,dev,test = generate_data()
numTrain = train[0].shape[0]
numTest = test[0].shape[0]
numDev = dev[0].shape[0]
print('Train/dev/test split: {}/{}/{}'.format(numTrain,numDev,numTest))

batchSize = 128
lstmUnits = 64
n_epochs = 5

timeSteps = 288
featureNo = 28

train_data = tf.data.Dataset.from_tensor_slices(train).map(input_parser)
train_data = train_data.shuffle(1000000)
train_data = train_data.apply(tf.contrib.data.batch_and_drop_remainder(batchSize))

iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)

X,Y = iterator.get_next()
train_init = iterator.make_initializer(train_data)
data = tf.cast(X, tf.float32)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype = tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, 1]))
bias = tf.Variable(tf.constant(0.1, shape=[1]))
value = tf.transpose(value, [1,0,2])
last = tf.gather(value, int(value.get_shape()[0]) -1)

prediction = (tf.matmul(last, weight) + bias)

error = tf.reduce_mean(tf.square(prediction - tf.cast(Y, tf.float32) ))
optimizer = tf.train.AdamOptimizer().minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    start_time = time.time()
    sess.run(init)

    for i in range(n_epochs):
        sess.run(train_init)
        total_error = 0
        n_batches = 0
        n_errors = 0
        try:
            while True:
                _, error_ = sess.run([optimizer, error])
                total_error += error_
                n_batches += 1
                print('epoch {}, batch {}, error: {}'.format(i, n_batches, error_))
        except tf.errors.OutOfRangeError:
            n_errors += 1
            pass
        print('----epoch {}---- avg error: {}'.format(i, total_error/n_batches))


    print('Total time: {0} seconds'.format(time.time() - start_time))
