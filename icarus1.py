import numpy as np
import datetime, time
import tensorflow as tf
import sys, os, re, gzip, json
from sklearn.utils import shuffle

INPUT_TENSOR_NAME = 'inputs'



def load_json(file_name):
    with open(file_name + '.json', 'rb') as fin:
        json_bytes = fin.read()
        json_str = json_bytes.decode('utf-8')
        data = json.loads(json_str)
        return data

data = np.load('data/btcusd_dataset.npy')

test_mode = True
if test_mode:
    data = data[-2000:]

def generate_data(window_size = 288, step_size = 1, dev_size = 0.2, test_size = 0.01):

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


def serving_input_receiver_fn():
    receiver_tensors = {
    INPUT_TENSOR_NAME: tf.placeholder(tf.int32, [None, 288, 28])
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=receiver_tensors, features=receiver_tensors)


def model_fn(features, labels, mode, params):
    X = features[INPUT_TENSOR_NAME]
    Y = labels

    batchSize = params['batchSize']
    lstmUnits = params['lstmUnits']
    timeSteps = params['timeSteps']
    featureNo = params['featureNo']
    pow_s = params['pow_s']
    pow_e = params['pow_e']
    tot_step_count = params['tot_step_count']

    data = tf.cast(X, tf.float32)
    lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype = tf.float32)

    weight = tf.Variable(tf.truncated_normal([lstmUnits, 1]))
    bias = tf.Variable(tf.constant(0.1, shape=[1]))
    value = tf.transpose(value, [1,0,2])
    last = tf.gather(value, int(value.get_shape()[0]) -1)

    prediction = (tf.matmul(last, weight) + bias)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'values': prediction}
        export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, export_outputs=export_outputs)


    global_step_ = tf.train.get_global_step()
    lambda_ = pow_s - (tf.cast(global_step_, tf.float32) * (pow_s - pow_e)) / (tot_step_count)
    tf.summary.scalar('lambda', lambda_, family = 'one')

    error = tf.reduce_mean(tf.square(prediction - tf.cast(Y, tf.float32) ))
    error = tf.pow(error, lambda_)

    if mode == tf.estimator.ModeKeys.EVAL:
        mse = tf.metrics.mean_squared_error(labels = tf.cast(Y, tf.float32) , predictions = prediction, name = 'mse_opp')
        metrics = {'eval_mse': mse}
        tf.summary.scalar('eval_mse', mse, family='one')
        return tf.estimator.EstimatorSpec(mode, loss=error, eval_metric_ops=metrics)

    if mode == tf.estimator.ModeKeys.TRAIN:
        # tf.summary.scalar('train_mse', error, family='one')
        train_op = tf.train.AdamOptimizer().minimize(error, global_step = tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=error, train_op=train_op)



def input_fn(input_data, batch_size, is_training, is_pred=False):
    dataset = tf.data.Dataset.from_tensor_slices(input_data).map(input_parser)

    if is_training:
        dataset = dataset.shuffle(6000)#.repeat().batch(batch_size)

    if is_pred:
        dataset = dataset.batch(1)
    else:
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    X,Y = dataset.make_one_shot_iterator().get_next()
    return ({INPUT_TENSOR_NAME:X}, Y)



def main():
    train,dev,test = generate_data()
    numTrain = int(train[0].shape[0])
    numTest = int(test[0].shape[0])
    numDev = int(dev[0].shape[0])
    print('Train/dev/test split: {}/{}/{}'.format(numTrain,numDev,numTest))


    epochs = 1
    batch_size = 128
    tot_step_count = (numTrain / batch_size)*epochs
    print('Total steps: {}'.format(tot_step_count))

    params = {
    'lstmUnits': 64,
    'batchSize': 148,
    'timeSteps': 288,
    'featureNo': 28,
    'pow_s': 3,
    'pow_e': 1,
    'tot_step_count': round(tot_step_count)
    }


    t0=time.time()
    training_config = tf.estimator.RunConfig(
    model_dir = '.\\models\\',
    save_summary_steps=50,
    save_checkpoints_steps=10000
    )
    icarus_classifier = tf.estimator.Estimator(model_fn = model_fn, params=params, config = training_config)


    for _ in range(epochs):
        icarus_classifier.train(
        input_fn = lambda: input_fn(train, batch_size, is_training=True),
        steps = None
        )

        # eval_input_fn = input_fn(test, 128, is_training = False)
        eval_result = icarus_classifier.evaluate(
        input_fn = lambda: input_fn(dev, batch_size, is_training=False)
        )

        print(eval_result)




    print('completed in {} seconds. '.format(time.time()-t0))


    print('-'*80)
    print(test[0].shape[0])
    print('-'*80)

    predictions = icarus_classifier.predict(
    input_fn = lambda: input_fn(test, 128, is_training=False, is_pred=True)
    )
    preds = [item for item in predictions]
    print(test[0].shape[0], len(preds))

    saved_dir = ".\serving_model"
    icarus_classifier.export_savedmodel(saved_dir, serving_input_receiver_fn=serving_input_receiver_fn)

if __name__=='__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()
