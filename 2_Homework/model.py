import tensorflow as tf
import numpy as np

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
    
    
    
    with tf.name_scope('two_layer_simple_conv_net') as scope:
        # first hidden conv layer
        hidden_1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu, name='hidden_1')
        
        # first pooling layer
        pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        
        # second hidden conv layer
        hidden_2 = tf.layers.conv2d(inputs=pool_1, filters=64, kernel_size=5, padding='same', activation=tf.nn.relu, name='hidden_2')
        
        # second pooling layer
        pool_2 = tf.layers.max_pooling2d(hidden_2, 2, 2, padding='same')
        
        # flatten from 4D to 2D for dense layer
        flat = tf.reshape(pool_2, [-1, 8*8*64])
        
        # dense layer output
        output = tf.layers.dense(flat, 100, name='output_layer')
        print('Output: {}'.format(output))
        
    tf.identity(output, name='output')

    
    # get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs= opt_metrics(y, output, lr, b_1, b_2)

    return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs


def opt_metrics(labels, predictions, learning_rate, beta_1, beta_2):
    
    
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
    rhs = classification_error + 1.96 *( np.sqrt((classification_error * (accuracy))/labels.shape[0]))
    lhs = classification_error - 1.96 *( np.sqrt((classification_error * (accuracy))/labels.shape[0])) 

    # saver
    saver = tf.train.Saver()

    return confusion_matrix_op, xentropy_w_reg, train_op, global_step_tensor, saver, accuracy, lhs, rhs