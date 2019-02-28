import numpy as np
import os
import argparse
import tensorflow as tf
from util import one_hot_encode, split_data
from model import model_summary, autoencoder, FourLayerConvNet

# setup parser
parser = argparse.ArgumentParser(description='Classify Fmnist images.')
parser.add_argument(
    '--data_dir',
    type=str,
    default='/work/cse496dl/shared/homework/02',
    help='directory where CIFAR-100 is located')
parser.add_argument(
    '--model_dir',
    type=str,
    default='/work/netthinker/ayush/2_hw_logs/',
    help='directory where model graph and weights are saved')
parser.add_argument('--batch_size', type=int, default=32, help='mini batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to run')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for Adam')
parser.add_argument('--filter_1', type=int, default=32, help='Filter for hidden conv layer')
parser.add_argument('--filter_2', type=int, default=64, help='Filter for hidden conv layer')
parser.add_argument('--kernel_size', type=int, default=5, help='Kernel size for simple conv layer')
parser.add_argument('--momentum_1', type=float, default=0.99, help='Adam hyperparameter')
parser.add_argument('--momentum_2', type=float, default=0.999, help='Adam hyperparameter')
parser.add_argument('--numOfLayers', type=int, default=2, help='number of layers in the model. 2 or 4')
parser.add_argument('--reg_scale', type=float, default=0.00, help='Scale for the regularization. Between 0.0 - 0.1')
parser.add_argument('--load_data', type=str, default='CIFAR')

# setup parser arguments
args = parser.parse_args()


# tensorflow placeholders for data and labels
input = tf.placeholder(tf.float32, [None, 32, 32, 3], name='input_placeholder')
output = tf.placeholder(tf.float32, [None, 100], name='label_placeholder')
                
# layer size
layer_size_1=256
layer_size_2=256

session = tf.Session()

if args.load_data == 'CIFAR':
    # get CIFAR-100 data and process it
    labels = np.load('/work/cse496dl/shared/homework/02/cifar_labels.npy')
    data = np.load('/work/cse496dl/shared/homework/02/cifar_images.npy')
    data = np.reshape(data, [-1, 32, 32, 3])

    # one hot encode labels
    labels = one_hot_encode(labels)

    # train-test split
    train_data, train_labels, test_data, test_labels = split_data(0.9, data, labels)

    # number of examples
    train_num_examples, test_num_examples = train_data.shape[0], test_data.shape[0]

    # four layer
    confusion_matrix_op_1, cross_entropy_1, train_op_1, global_step_tensor_1, saver_1, accuracy_1, lhs_1, rhs_1 = FourLayerConvNet(input, output, args.filter_1, args.filter_2, args.kernel_size, args.learning_rate, args.momentum_1, args.momentum_2)
    
    print("LABELS")
    print(labels.shape)
    print("=======================================")
    print("DATA SHAPE")
    print(data.shape)

else:
    # get ImageNet data and process its
    data = np.load('/work/cse496dl/shared/homework/02/imagenet_images.npy')
    train_data = np.reshape(data, [-1, 32, 32, 3])
    train_num_examples = train_data.shape[0]

    # image net 
    total_loss_1, train_op_1, global_step_tensor_1, saver_1 = autoencoder(input, args.learning_rate, args.momentum_1, args.momentum_2)

    print("=======================================")
    print("DATA SHAPE")
    print(data.shape)



# Training
with session as session:
    # initialize variables
    session.run(tf.global_variables_initializer())
    
    # batch size
    batch_size = args.batch_size
    
    # Hypterparameter information
    print('Data: {}'.format(args.load_data))
    print('Model: {}'.format('Autoencoder'))
    print('Batch Size: {}'.format(batch_size))
    print('Epochs: {}'.format(args.epochs))
    print('Filter 1: {}'.format(args.filter_1))
    print('Filter 2: {}'.format(args.filter_2))
    print('Kernel Size: {}'.format(args.kernel_size))
    model_summary()

    # training loop
    for epoch in range(args.epochs):
        
        # get current epoch
        print('Epoch: {}'.format(str(epoch)))
        
        # list to store cross entropy and confusion matrices
        loss_vals = []
        conf_mxs = []
        
        # run gradient steps tnd report mean loss on train data
        for i in range(train_num_examples // batch_size):
            
            # train
            if args.load_data == 'CIFAR':
                # get batches of data
                batch_xs = train_data[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = train_labels[i * batch_size:(i+1) * batch_size, :]
                _, train_loss, conf_matrix, accuracy, lhs, rhs = session.run([train_op_1, cross_entropy_1, 
                                                                                confusion_matrix_op_1, accuracy_1, lhs_1, rhs_1], {input: batch_xs, output: batch_ys})
                # append cross entropy loss and confusion matrix predictions
                loss_vals.append(train_loss)
                conf_mxs.append(conf_matrix)

            else:
                # get batches of data
                batch_xs = train_data[i * batch_size:(i + 1) * batch_size, :]
                _, train_loss = session.run([train_op_1, total_loss_1], {input: batch_xs})
                # append loss
                loss_vals.append(train_loss) 

        avg_train_loss = sum(loss_vals) / len(loss_vals)        

        if args.load_data == 'CIFAR':        
            # return metrics
            print('----------TRAIN STATS----------\n')
            print('TRAINING LOSS: ' + str(avg_train_loss))
            print('TRAIN ACCURACY: ' + str(accuracy))
            print('TRAIN CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))
            print('TRAIN CONFIDENCE INTERVAL: {},{}\n'.format(lhs,rhs))
        else:
             # return metrics
            print('----------TRAIN STATS----------\n')
            print('TRAINING LOSS: ' + str(min(avg_train_loss)))
    
        # test
        if args.load_data == 'CIFAR':

            print('----------TEST STATS----------\n')
            # resetting variables
            ce_vals = []
            conf_mxs = []
            
            # run test
            for i in range(test_num_examples // batch_size):
                
                # get batches of data
                batch_xs = test_data[i * batch_size:(i + 1) * batch_size, :]
                batch_ys = test_labels[i * batch_size:(i + 1) * batch_size, :]
                
                # train
                test_ce, conf_matrix_t, accuracy_t, lhs_t, rhs_t= session.run([cross_entropy_1, confusion_matrix_op_1, accuracy_1, lhs_1, rhs_1],{input: batch_xs, output: batch_ys})
                
                # append cross entropy loss and confusion matrix predictions
                ce_vals.append(test_ce)
                conf_mxs.append(conf_matrix_t)

            # test cross entropy
            avg_test_ce = sum(ce_vals) / len(ce_vals)

            # return metrics
            print('TEST CROSS ENTROPY: ' + str(avg_test_ce))
            print('TEST ACCURACY: ' + str(accuracy_t))
            print('TEST CONFUSION MATRIX:')
            print(str(sum(conf_mxs)))
            print('TEST CONFIDENCE INTERVAL: {},{}'.format(lhs_t, rhs_t))

    # model files saver
    path_prefix = saver_1.save(session, args.model_dir + "homework_2")

        
print('----------FINISHED----------\n')