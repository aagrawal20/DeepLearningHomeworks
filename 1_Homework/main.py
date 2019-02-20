from util import one_hot_encode, split_data
from model import TwoLayerNet, FourLayerNet
import tensorflow as tf
import numpy as np
import os
import argparse

# setup parser
parser = argparse.ArgumentParser(description='Classify Fmnist images.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/work/cse496dl/shared/homework/01',
    help='directory where Fmnist is located')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/work/netthinker/ayush/new_hw_logs/',
    help='directory where model graph and weights are saved')
parser.add_argument(
    '--log_dir',
    type=str,
    default='/work/netthinker/ayush/new_logs/1/',
    help='directory where tensorboard logs are saved')
parser.add_argument('--batch_size', type=int, default=32, help='mini batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to run')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for Adam')
parser.add_argument('--momentum_1', type=float, default=0.99, help='Adam hyperparameter')
parser.add_argument('--momentum_2', type=float, default=0.999, help='Adam hyperparameter')
parser.add_argument('--numOfLayers', type=int, default=2, help='number of layers in the model. 2 or 4')
parser.add_argument('--reg_scale', type=float, default=0.00, help='Scale for the regularization. Between 0.0 - 0.1')

# setup parser arguments
args = parser.parse_args()


# get data and process it
labels = np.load('/work/cse496dl/shared/homework/01/fmnist_train_labels.npy')
data = np.load('/work/cse496dl/shared/homework/01/fmnist_train_data.npy')

# one hot encode labels
labels = one_hot_encode(labels)

# train-test split
train_data, train_labels, test_data, test_labels = split_data(0.9, data, labels)

# get number of examples
train_num_examples, test_num_examples = train_data.shape[0], test_data.shape[0]

# create tensorflow placeholders for data and labels
input = tf.placeholder(tf.float32, [None, 784], name='input_placeholder')
output = tf.placeholder(tf.float32, [None, 10], name='label_placeholder')

# layer size
layer_size_1=256
layer_size_2=256

# get the model based on argument
if args.numOfLayers is 2:   
    # two layer                
    confusion_matrix_op_1, cross_entropy_1, train_op_1, global_step_tensor_1, saver_1, accuracy_1, merge_1 = TwoLayerNet(input, output, args.learning_rate, args.momentum_1, args.momentum_2, layer_size_1, layer_size_2, args.reg_scale)
else:
    #four layer
    confusion_matrix_op_1, cross_entropy_1, train_op_1, global_step_tensor_1, saver_1, accuracy_1, merge_1 = FourLayerNet(input, output, args.learning_rate, args.momentum_1, args.momentum_2, layer_size_1, layer_size_2, args.reg_scale)                 

#Training
with tf.Session() as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    
    # writer for tensorboard
    train_writer = tf.summary.FileWriter( args.log_dir + 'train', session.graph)
    
    # batch size
    batch_size = args.batch_size
    
    # Hyperparmater information
    print('Batch Size: {}'.format(batch_size))
    print('Epochs: {}'.format(args.epochs))
    print('Number of layers: {}'.format(args.numOfLayers))
    print('Learning Rate: {}'.format(args.learning_rate))
    print('Layer Size: {}'.format(layer_size_1))
    print('Reg Scale: {}'.format(args.reg_scale))

    # counter for tensorboard
    counter = 0

    # training loop
    for epoch in range(args.epochs):
        
        # get current epoch
        print('Epoch: ' + str(epoch))

        # list to store cross entropy and confusion matrices
        ce_vals = []
        conf_mxs = []

        # run gradient steps and report mean loss on train data
        for i in range(train_num_examples // batch_size):
            
            # tensorboard stuff
            counter +=1
            
            # get batches of data
            batch_xs = train_data[i * batch_size:(i + 1) * batch_size, :]
            batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
            
            # train
            summary, _, train_ce, conf_matrix, accuracy = session.run([merge_1, train_op_1, cross_entropy_1, confusion_matrix_op_1, accuracy_1], {input: batch_xs, output: batch_ys})
            
            # tensorboard 
            train_writer.add_summary(summary, counter)
            
            # append cross entropy loss and confusion matrix predictions
            ce_vals.append(train_ce)
            conf_mxs.append(conf_matrix)
         
        # classification error
        classification_error = 1 - accuracy


        # confidence interval
        rhs = classification_error + 1.96 *( np.sqrt((classification_error * (accuracy))/train_labels.shape[0]))
        lhs = classification_error - 1.96 *( np.sqrt((classification_error * (accuracy))/train_labels.shape[0])) 
        
        # train crossentropy
        avg_train_ce = sum(ce_vals) / len(ce_vals)
        
        # return metrics
        print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
        print('TRAIN ACCURACY: ' + str(accuracy))
        print('TRAIN CONFUSION MATRIX:')
        print(str(sum(conf_mxs)))
        print('TRAIN CONFIDENCE INTERVAL: {},{}'.format(lhs,rhs))


    # Test

    # counter for tensorboard
    counter = 0  
    
    # resetting variables
    ce_vals = []
    conf_mxs = []

    # writer for test
    test_writer = tf.summary.FileWriter( args.log_dir + 'test', session.graph)

    # run test
    for i in range(test_num_examples // batch_size):

        # tensorboard stuff
        counter += 1
        
        # get batches of data
        batch_xs = test_data[i * batch_size:(i + 1) * batch_size, :]
        batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
        
        # train
        summary, _, test_ce, conf_matrix_t, accuracy_t= session.run([merge_1, train_op_1, cross_entropy_1, confusion_matrix_op_1, accuracy_1],{input: batch_xs, output: batch_ys})

        #tensorboard
        test_writer.add_summary(summary, counter)
        
        # append cross entropy loss and confusion matrix predictions
        ce_vals.append(test_ce)
        conf_mxs.append(conf_matrix_t)

    # classification error
    classification_error = 1 - accuracy_t

    # confidence interval
    rhs = classification_error + 1.96 * (np.sqrt((classification_error * (accuracy_t)) / train_labels.shape[0]))
    lhs = classification_error - 1.96 * (np.sqrt((classification_error * (accuracy_t)) / train_labels.shape[0]))

    # test crossentropy
    avg_test_ce = sum(ce_vals) / len(ce_vals)

    # retunr metrics
    print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
    print('TEST ACCURACY: ' + str(accuracy_t))
    print('TEST CONFUSION MATRIX:')
    print(str(sum(conf_mxs)))
    print('TEST CONFIDENCE INTERVAL: {},{}'.format(lhs, rhs))

    # model files saver
    path_prefix = saver_1.save(session, args.model_dir + "homework_1")

print("========================================================")
print("FINISHED!")
print("========================================================")
