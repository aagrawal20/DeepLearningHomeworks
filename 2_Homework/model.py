import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim


def conv_block(inputs, filters):
    with tf.name_scope('two_layer_simple_conv_net') as scope:
        # first hidden conv layer
        hidden_1 = tf.layers.Conv2D(filters=filters[0], kernel_size=5, padding='same', activation=tf.nn.relu, name='hidden_1')
        
        # first pooling layer
        # pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        
        # second hidden conv layer
        hidden_2 = tf.layers.Conv2D(filters=filters[1], kernel_size=5, padding='same', activation=tf.nn.relu, name='hidden_2')
        
        output_tensor =hidden_2(hidden_1(inputs))
        layer_list = [hidden_1, hidden_2]

        block_parameter_num = sum(map(lambda layer: layer.count_params(), layer_list))
        print('Number of parameters in conv block with {} input: {}'.format(inputs.shape, block_parameter_num))
        return output_tensor


def FourLayerConvNet(x, y, f_1, f_2, k_size, lr, b_1, b_2):
        """
        Args:
                - x: models predicted output
                - y: Actual labels for each example        
                - f_1: filters for hidden layer 1
                - f_2: filter for hidden layer 2
                - lr: rate for the Adam optimizer
                - b_1: first momentum for Adam
                - b_2: second momentum for Adam
        """
        pool = tf.layers.MaxPooling2D(2, 2, padding='same')

        conv_block_1 = pool(conv_block(x,[32, 64]))

        dropout_1 = tf.layers.dropout(conv_block_1, rate=0.3, name='dropout_1')

        conv_block_2 = pool(conv_block(dropout_1, [64,128]))

        dropout_2 = tf.layers.dropout(conv_block_2, rate=0.3, name='dropout_2')

        # flatten from 4D to 2D for dense layer
        flat = tf.reshape(dropout_2, [-1, 8*8*128])

        # dense layer output
        output = tf.layers.dense(flat, 100, name='output_layer')

        tf.identity(output, name='output')
        # get the model metrics
        confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs= opt_metrics(x, y, output, lr, b_1, b_2)

        return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs


def TwoLayerSimpleConvNet(x, y, f_1, f_2, k_size, l_size_1, l_size_2, lr, b_1, b_2):
        """
                Args:
                - x: models predicted output
                - y: Actual labels for each example
                - lr: rate for the Adam optimizer
                - b_1: first momentum for Adam
                - b_2: second momentum for Adam
                - f_1: filters for hidden layer 1
                - f_2: filter for hidden layer 2
                - l_size_1: layer size for hidden layers
                - l_size_2: another layer size for hidden layers
        """
        pool = tf.layers.MaxPooling2D(2, 2, padding='same')

        conv_x = pool(conv_block(x, [32,64]))
                
        # flatten from 4D to 2D for dense layer
        flat = tf.reshape(conv_x, [-1, 16*16*64])

        # dense layer output
        output = tf.layers.dense(flat, 100, name='output_layer')

        tf.identity(output, name='output')
        # get the model metrics
        confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs= opt_metrics(x, y, output, lr, b_1, b_2)

        return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs


def opt_metrics(x, labels, predictions, learning_rate, beta_1, beta_2):
    
        # cross entropy loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions))

        # collect the regularization losses
        regularization_losses= tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        # weight of the regularization for the final loss
        REG_COEFF = 0.1

        # value passes to minimize
        xentropy_w_reg = cross_entropy + REG_COEFF * sum(regularization_losses)

        # confusion matrix
        confusion_matrix_op  = tf.confusion_matrix(tf.argmax(labels, axis=1), tf.argmax(predictions, axis=1), num_classes=100)

        # training and saving functionality
        global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)

        # Adam
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2)

        # optimizer
        train_op = optimizer.minimize(xentropy_w_reg, global_step=global_step_tensor)

        # correct predictions
        correct = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))

        # accuracy
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # classifiation
        classification_error = 1 - accuracy

        # confidence interval
        rhs = classification_error + 1.96 *( tf.sqrt((classification_error * (accuracy))/tf.cast(tf.shape(labels, out_type=tf.int32), tf.float32)))
        lhs = classification_error - 1.96 *( tf.sqrt((classification_error * (accuracy))/tf.cast(tf.shape(labels, out_type=tf.int32), tf.float32)))

        # saver
        saver = tf.train.Saver()

        return confusion_matrix_op, xentropy_w_reg, train_op, global_step_tensor, saver, accuracy, lhs, rhs

def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)