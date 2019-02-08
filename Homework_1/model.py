import tensorflow as tf

def opt_metrics(labels, predictions, learning_rate, beta_1, beta_2):
    """
    Args:
        - labels: Actual labels for each example
        - predictions: models predicted output
        - learning rate: rate for the Adam optimizer
        - beta 1: first momentum for Adam
        - beta 2: second momentum for Adam
    """
    # cross entropy loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=output))

    # collect the regularization losses
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    # this is the weight of the regularization part of the final loss
    REG_COEFF = 0.1
    # this value is what we'll pass to `minimize`
    xentrop_w_reg = cross_entropy + REG_COEFF * sum(regularization_losses)

    # confusion matrix
    confusion_matrix_op = tf.confusion_matrix(tf.argmax(y, axis=1), tf.argmax(output, axis=1), num_classes=10)

    # set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)

    # Adam
    optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=b_1, beta2=b_2)

    # optimizer
    train_op = optimizer.minimize(xentrop_w_reg, global_step=global_step_tensor)

    # correct predictions
    correct = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))

    # accuracy
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    # saver
    saver = tf.train.Saver()

    return confusion_matrix, xentrop_w_reg, train_op, global_step_tensor, saver, accuracy


def OneLayerNet(x, y, lr, b_1, b_2, l_size_1, l_size_2, reg_scale):
     """
     Args:
        - x: models predicted output
        - y: Actual labels for each example
        - lr: rate for the Adam optimizer
        - b_1: first momentum for Adam
        - b_2: second momentum for Adam
        - l_size_1: layer size for hidden layers
        - l_size_2: another layer size for hidden layers
        - reg: scale 0.0 for no regularization
    """
    # normalize data
    x = x / 255.0

    # model
    with tf.name_scope('one_layer_net') as scope:

        # hidden layer with L2
        hidden = tf.layers.dense(x, l_size_1, activation=tf.nn.relu,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 name='hidden_layer_1')

        # dropout layer
        dropout = tf.layers.dropout(hidden, rate=0.2, name='dropout_1')

        # output layer with L2
        output = tf.layers.dense(dropout, 10,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 name='output_layer')

    tf.identity(output, name='output')

    # get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy = opt_metrics(y, output, lr, b_1, b_2)
    
    return confusion_matrix_op, cross_entropy, train_op, global_step_tensor, saver,accuracy


def TwoLayerNet(x, y, lr, b_1, b_2, l_size_1, l_size_2, reg_scale):
    """
     Args:
        - x: models predicted output
        - y: Actual labels for each example
        - lr: rate for the Adam optimizer
        - b_1: first momentum for Adam
        - b_2: second momentum for Adam
        - l_size_1: layer size for hidden layers
        - l_size_2: another layer size for hidden layers
        - reg: scale 0.0 for no regularization
    """
    
    # normalize data
    x = x / 255.0

    # model
    with tf.name_scope('two_layer_net') as scope:

        # first hidden layer with L2
        hidden_1 = tf.layers.dense(x, l_size_1, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_1')

        # second hidden layer with L2
        hidden_2 = tf.layers.dense(hidden_1, l_size_1, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_2')

        # output layer with L2
        output = tf.layers.dense(hidden_2, 10,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 name='output_layer')

    tf.identity(output, name='output')

     # get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy = opt_metrics(y, output, lr, b_1, b_2)

    return confusion_matrix_op, cross_entropy, train_op, global_step_tensor, saver, accuracy


def ThreeLayerNet(x, y, lr, b_1, b_2, l_size_1, l_size_2, reg_scale):
    """
     Args:
        - x: models predicted output
        - y: Actual labels for each example
        - lr: rate for the Adam optimizer
        - b_1: first momentum for Adam
        - b_2: second momentum for Adam
        - l_size_1: layer size for hidden layers
        - l_size_2: another layer size for hidden layers
        - reg: scale 0.0 for no regularization
    """
    
    # normalize data
    x = x / 255.0

    # model
    with tf.name_scope('two_layer_net') as scope:

        # first hidden layer with L2
        hidden_1 = tf.layers.dense(x, l_size_1, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_1')


        # second hidden layer with L2
        hidden_2 = tf.layers.dense(hidden_1, l_size_1, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_2')
        

        # third hidden layer with L2
        hidden_3 = tf.layers.dense(hidden_2, l_size_1, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_3')


        # output layer with L2
        output = tf.layers.dense(hidden_3, 10,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                 name='output_layer')


    tf.identity(output, name='output')

    # get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy = opt_metrics(y, output, lr, b_1, b_2)

    return confusion_matrix_op, xentrop_w_reg, train_op, global_step_tensor, saver, accuracy


def FourLayerNet(x, y, lr, b_1, b_2, l_size_1, l_size_2, reg_scale):
    """
     Args:
        - x: models predicted output
        - y: Actual labels for each example
        - lr: rate for the Adam optimizer
        - b_1: first momentum for Adam
        - b_2: second momentum for Adam
        - l_size_1: layer size for hidden layers
        - l_size_2: another layer size for hidden layers
        - reg: scale 0.0 for no regularization
    """
    
    # normalize data
    x = x / 255.0

    # model
    with tf.name_scope('four_layer_net') as scope:

        # first hidden layer with L2
        hidden_1 = tf.layers.dense(x, l_size_1, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_1')

        # first dropout layer
        dropout_1 = tf.layers.dropout(hidden_1, rate=0.1, name='dropout_1')

        # second hidden layer with L2
        hidden_2 = tf.layers.dense(hidden_1, l_size_1, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_2')
        # second dropout layer
        dropout_2 = tf.layers.dropout(hidden_2, rate=0.1, name='dropout_2')

       # third hidden layer with L2
        hidden_3 = tf.layers.dense(hidden_2, l_size_2, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_3')
        # third dropout layer
        dropout_3 = tf.layers.dropout(hidden_3, rate=0.1, name='dropout_3')

        # fourth hidden layer with L2
        hidden_4 = tf.layers.dense(hidden_3, l_size_2, activation=tf.nn.relu,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(scale=reg_scale),
                                   name='hidden_layer_4')

        # fourth dropout layer
        dropout_4 = tf.layers.dropout(hidden_4, rate=0.1, name='dropout_4')

        # output layer with L2
        output = tf.layers.dense(dropout_4, 10,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01),
                                 name='output_layer')

    tf.identity(output, name='output')

    # get the model metrics
    confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy = opt_metrics(y, output, lr, b_1, b_2)

    return confusion_matrix_op, xentrop_w_reg, train_op, global_step_tensor, saver, accuracy