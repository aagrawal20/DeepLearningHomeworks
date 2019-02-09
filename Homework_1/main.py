from util import one_hot_encode, split_data, generate_adam, cross_validate
from model import TwoLayerNet, FourLayerNet
from sklearn.cross_validation import train_test_split
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
    default='/work/netthinker/ayush/homework_1_logs',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=32, help='mini batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to run')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for Adam')
parser.add_argument('--momentum_1', type=float, default=0.99, help='Adam hyperparameter')
parser.add_argument('--momentum_2', type=float, default=0.999, help='Adam hyperparameter')
parser.add_argument('--numOfLayers', type=int, default=2, help='number of layers in the model. 2 or 4')
parser.add_argument('--reg_scale', type=float, default=0.00, help='Scale for the regularization. Between 0.0 - 0.1')

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

# set hyperparameters
# b_1 = 0.99
# b_2 = 0.999
# learning_rate = 0.0001
layer_size_1 = 256
layer_size_2 = 256

# get the one layer model
if args.numOfLayers is 2:
                    
    confusion_matrix_op_1, cross_entropy_1, train_op_1, global_step_tensor_1, saver_1, accuracy_1 = TwoLayerNet(input, output, args.learning_rate, args.momentum_1, args.momentum_2, layer_size_1, layer_size_2, args.reg_scale)
else:
    confusion_matrix_op_1, cross_entropy_1, train_op_1, global_step_tensor_1, saver_1, accuracy_1 = FourLayerNet(input, output, args.learning_rate, args.momentum_1, args.momentum_2, layer_size_1, layer_size_2, args.reg_scale)                 

# init = tf.global_variables_initializer()

# session start
# with tf.Session() as session:
#     print("Momentum 1: ", b_1)
#     print("Momentum 2: ", b_2)
#     print("Learning Rate: ", learning_rate)
#     print("Layer size 1: ", layer_size_1)
#     print("Layer size 2: ", layer_size_2)
#     result = cross_validate(session, train_data, train_labels, cross_entropy_1, input, output, init, train_op_1, accuracy)
#     print ("Cross-validation result: %s" % result)    
#     saver_1.save(session, "/work/netthinker/ayush/homework_1_logs/homework_1")
#Training
with tf.Session() as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    
    train_writer = tf.summary.FileWriter( '/work/netthinker/ayush/logs/1/train ', session.graph)
    
    # batch size
    batch_size = args.batch_size
    
    print('Batch Size: {}'.format(batch_size))
    print('Epochs: {}'.format(args.epochs))
    print('Number of layers: {}'.format(args.numOfLayers))
    print('Learning Rate: {}'.format(args.learning_rate))
    print('Layer Size: {}'.format(layer_size_1))

    # val_min_loss = 999
    # no_loss_drop = 0
    # training loop
    for epoch in range(args.epochs):
        print('Epoch: ' + str(epoch))
        # list to store cross entropy
        ce_vals = []
        conf_mxs = []
        # run gradient steps and report mean loss on train data
        for i in range(train_num_examples // batch_size):
            # get batches of data
            batch_xs = train_data[i * batch_size:(i + 1) * batch_size, :]
            batch_ys = train_labels[i * batch_size:(i + 1) * batch_size, :]
            # train
            _, train_ce, conf_matrix, accuracy = session.run([train_op_1, cross_entropy_1, confusion_matrix_op_1, accuracy_1],
                                                   {input: batch_xs, output: batch_ys})
            # append cross entropy loss and confusion matrix predictions
            ce_vals.append(train_ce)
            conf_mxs.append(conf_matrix)

        avg_train_ce = sum(ce_vals) / len(ce_vals)
        print('TRAIN CROSS ENTROPY: ' + str(avg_train_ce))
        print('TRAIN ACCURACY: ' + str(accuracy))
        print('TRAIN CONFUSION MATRIX:')
        print(str(sum(conf_mxs)))

        # # VALIDATION
        # ce_vals = []
        # # run gradient steps and report mean loss on train data
        # for i in range(val_num_examples // batch_size):
        #     # get batches of data
        #     batch_xs = val_data[i * batch_size:(i + 1) * batch_size, :]
        #     batch_ys = val_labels[i * batch_size:(i + 1) * batch_size, :]
        #     # train
        #     _, val_ce = session.run([train_op_1, cross_entropy_1],
        #                             {input: batch_xs, output: batch_ys})
        #     # append cross entropy loss and confusion matrix predictions
        #     ce_vals.append(val_ce)
        #
        # avg_val_ce = sum(ce_vals) / len(ce_vals)
        # print('VAL CROSS ENTROPY: ' + str(avg_val_ce))
        #
        # # Implement Early stopping
        #
        # # assign min loss if not then increment counter
        # if val_min_loss > min(ce_vals):
        #     val_min_loss = min(ce_vals)
        #     no_loss_drop = 0
        #     # saver to save model
        #     # saver_1.save(
        #     #     session,
        #     #     os.path.join(args.model_dir, "homework_1"),
        #     #     global_step=global_step_tensor_1)
        #
        # else:
        #     no_loss_drop += 1
        #
        # # if no loss drop recorded in the past 5 epochs then break
        # if no_loss_drop > 5:
        #     print("Early Stopping the model training as the loss has not dropped in the last %d runs", no_loss_drop)
        #     break

        # # TESTING
        # ce_vals = []
        # conf_mxs = []
        # for i in range(test_num_examples // batch_size):
        #     batch_xs = test_data[i * batch_size:(i + 1) * batch_size, :]
        #     batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
        #     test_ce, conf_matrix = session.run([cross_entropy_1, confusion_matrix_op_1],
        #                                        {input: batch_xs, output: batch_ys})
        #     ce_vals.append(test_ce)
        #     conf_mxs.append(conf_matrix)
        #
        # avg_test_ce = sum(ce_vals) / len(ce_vals)
        #
        # print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
        # print('TEST CONFUSION MATRIX:')
        # print(str(sum(conf_mxs)))
    session.run([cross_entropy_1, accuracy_1], feed_dict={input: test_data, output: test_labels})
    path_prefix = saver_1.save(session, "/work/netthinker/ayush/homework_1_logs/homework_1")
    # saver_1.save(session, "./homework_1")
print("========================================================")
print("FINISHED!")
print("========================================================")
