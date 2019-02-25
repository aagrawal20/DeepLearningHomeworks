import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

# your code here
def upscale_block(x, scale=2):
    """transpose convolution upscale"""
    return tf.layers.conv2d_transpose(x, 3, 3, strides=(scale, scale), padding='same', activation=tf.nn.relu)

def downscale_block(x, scale=2):
    n, h, w, c = x.get_shape().as_list()
    return tf.layers.conv2d(x, 3, 3, strides=scale, padding='same')
# np.floor(c * 1.25)

def autoencoder(x, lr, b_1, b_2):
        """
        args: 
                - x: image batch of shape [?, 32, 32, 3]
                - lr: learning rate
                - b_1: momentum 1
                - b_2: momentum 2
        
        returns: 

        """
        
        # encode 1 = [-1, 32, 32, 32]
        encoder_1 = tf.layers.conv2d(x, 32, 3, strides=1, padding='same', name='encode_1')
        print('Encoder_1: {}'.format(encoder_1))
        
        # encode 2 = [-1, 16, 16, 64]
        encoder_2 = tf.layers.conv2d(encoder_1, 64, 3, strides=2, padding='same', name='encode_2')
        print('Encoder_2: {}'.format(encoder_2))
        
        # encode 3 = [-1, 8, 8, 128]
        encoder_3 = tf.layers.conv2d(encoder_2, 128, 3, strides=2, padding='same', name='encode_3')
        print('Encoder_3: {}'.format(encoder_3))
        
        # flatten dimensions =  8192
        flatten_dim = np.prod(encoder_3.get_shape().as_list()[1:])
        print('flatten_dim: {}'.format(flatten_dim))
        
        # flat = [-1, 8192]
        flat = tf.reshape(encoder_3, [-1, flatten_dim], name='flat')
        print('Flat: {}'.format(flat))
        
        # code 
        code = tf.layers.dense(flat, 1000, activation=tf.nn.relu, name='code')
        print('Code: {}'.format(code))
        
        # hidden decoder = [-1, 8192]
        hidden_decoder = tf.layers.dense(code, 8192, activation=tf.nn.relu, name='hidden_decode')
        print('Hidden Decoder: {}'.format(hidden_decoder))
        
        # decoder_1 = [-1, 8, 8, 128]
        decoder_1 = tf.reshape(hidden_decoder, [-1, 8, 8, 128], name='decode_1')
        print('Decoder 1: {}'.format(decoder_1))
        
        # decoder_2 = [-1, 16, 16, 64]
        decoder_2 = tf.layers.conv2d_transpose(decoder_1, 64, 3, strides=(2, 2), padding='same', activation=tf.nn.relu, name='decode_2')
        print('Decoder 2: {}'.format(decoder_2))
        
        # decoder_3 = [-1, 32, 32, 32]
        decoder_3 = tf.layers.conv2d_transpose(decoder_2, 32, 3, strides=(2, 2), padding='same', activation=tf.nn.relu, name='decode_3')
        print('Decoder 3: {}'.format(decoder_3))
        
        # output = [-1, 32, 32, 3]
        output = tf.layers.conv2d_transpose(decoder_3, 3, 3, strides=(1,1), padding='same', activation=tf.nn.relu, name='output')
        print('Output: {}'.format(output))
        
        return opt_metrics_autoencoder(x, code, output, lr, b_1, b_2)
        

def conv_block(inputs, filters, name):
    with tf.name_scope(name) as scope:
        # first hidden conv layer
        hidden_1 = tf.layers.Conv2D(filters=filters[0], kernel_size=3, padding='same', activation=tf.nn.relu, name=name+'_1')
        
        # first pooling layer
        # pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        
        # second hidden conv layer
        hidden_2 = tf.layers.Conv2D(filters=filters[1], kernel_size=3, padding='same', activation=tf.nn.relu, name=name+'_2')
        pool = tf.layers.MaxPooling2D(2, 2, padding='same')
        output_tensor=pool(hidden_2(hidden_1(inputs)))
        layer_list = [hidden_1, hidden_2, pool]

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
        # pool = tf.layers.MaxPooling2D(2, 2, padding='same')

        conv_block_1 = conv_block(x,[32, 32], name='conv1')

        dropout_1 = tf.layers.dropout(conv_block_1, rate=0.3, name='dropout_1')

        conv_block_2 = conv_block(dropout_1, [64,64], name='conv2')

        dropout_2 = tf.layers.dropout(conv_block_2, rate=0.3, name='dropout_2')

        # flatten from 4D to 2D for dense layer
        flat = tf.reshape(dropout_2, [-1, 8*8*64])

        # dense layer 1
        dense_1 = tf.layers.dense(flat, 1024, name='dense_1', activation=tf.nn.relu)

        # dense layer 2
        dense_2 = tf.layers.dense(dense_1, 512, name='dense_2', activation=tf.nn.relu)

        # dense layer output
        output = tf.layers.dense(dense_2, 100, name='output_layer')

        tf.identity(output, name='output')
        # get the model metrics
        confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs= opt_metrics(x, y, output, lr, b_1, b_2)
        
        return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs


# def TwoLayerSimpleConvNet(x, y, f_1, f_2, k_size, l_size_1, l_size_2, lr, b_1, b_2):
#         """
#                 Args:
#                 - x: models predicted output
#                 - y: Actual labels for each example
#                 - lr: rate for the Adam optimizer
#                 - b_1: first momentum for Adam
#                 - b_2: second momentum for Adam
#                 - f_1: filters for hidden layer 1
#                 - f_2: filter for hidden layer 2
#                 - l_size_1: layer size for hidden layers
#                 - l_size_2: another layer size for hidden layers
#         """
#         pool = tf.layers.MaxPooling2D(2, 2, padding='same')

#         conv_x = pool(conv_block(x, [32,64], 'conv1'))
                
#         # flatten from 4D to 2D for dense layer
#         flat = tf.reshape(conv_x, [-1, 16*16*64])

#         # dense layer output
#         output = tf.layers.dense(flat, 100, name='output_layer')

#         tf.identity(output, name='output')
        
#         # get the model metrics
#         confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs= opt_metrics(x, y, output, lr, b_1, b_2)

#         return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs


def opt_metrics_autoencoder(x, code, output, learning_rate, beta_1, beta_2):
        
        # calculate loss
        sparsity_weight = 5e-3
        sparsity_loss = tf.norm(code, ord=1, axis=1)
        reconstruction_loss = tf.reduce_mean(tf.square(output - x)) # Mean Square Error
        total_loss = reconstruction_loss + sparsity_weight * sparsity_loss
        # total_loss = reconstruction_loss

        global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
        # Adam
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=beta_1, beta2=beta_2)

        # optimizer
        train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)

        saver = tf.train.Saver() 

        return total_loss, train_op, global_step_tensor, saver

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
