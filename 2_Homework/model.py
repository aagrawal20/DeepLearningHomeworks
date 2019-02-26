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
        encoder_1 = tf.layers.Conv2D(32, 3, 1, padding='same', name='encode_1', activation=tf.nn.relu)(x)
        
        # print('Encoder_1: {}'.format(encoder_1))
        
        # batch norm 1
        b_norm_1 = tf.layers.BatchNormalization(trainable=True)(encoder_1)
        # print('Batch Norm 1: {}'.format(b_norm_1))

        # encode 2 = [-1, 16, 16, 64]
        encoder_2 = tf.layers.Conv2D(64, 3, 2, padding='same', name='encode_2', activation=tf.nn.relu)(b_norm_1)
        # print('Encoder_2: {}'.format(encoder_2))

        # batch norm 2
        b_norm_2 = tf.layers.BatchNormalization(trainable=True)(encoder_2)
        # print('Batch Norm 2: {}'.format(b_norm_2))

        # encode 3 = [-1, 8, 8, 128]
        encoder_3 = tf.layers.Conv2D(128, 3, 2, padding='same', name='encode_3', activation=tf.nn.relu)(b_norm_2)
        # print('Encoder_3: {}'.format(encoder_3))

        # batch norm 3
        b_norm_3 = tf.layers.BatchNormalization(trainable=True)(encoder_3)
        # print('Batch Norm 3: {}'.format(b_norm_3))

        # flatten dimensions =  8192
        flatten_dim = np.prod(b_norm_3.get_shape().as_list()[1:])
        # print('flatten_dim: {}'.format(flatten_dim))
        
        # flat = [-1, 8192]
        flat = tf.reshape(b_norm_3, [-1, flatten_dim], name='flat')
        # print('Flat: {}'.format(flat))
        
        # code 
        code = tf.layers.Dense(1000, activation=tf.nn.relu, name='code')(flat)
        # print('Code: {}'.format(code))
        
        # hidden decoder = [-1, 8192]
        hidden_decoder = tf.layers.Dense(8192, activation=tf.nn.relu, name='hidden_decode')(code)
        # print('Hidden Decoder: {}'.format(hidden_decoder))
        
        # decoder_1 = [-1, 8, 8, 128]
        decoder_1 = tf.reshape(hidden_decoder, [-1, 8, 8, 128], name='decode_1')
        # print('Decoder 1: {}'.format(decoder_1))
        
        # decoder_2 = [-1, 16, 16, 64]
        decoder_2 = tf.layers.Conv2DTranspose(64, 3, (2, 2), padding='same', activation=tf.nn.relu, name='decode_2')(decoder_1)
        # print('Decoder 2: {}'.format(decoder_2))
        
        # decoder_3 = [-1, 32, 32, 32]
        decoder_3 = tf.layers.Conv2DTranspose(32, 3, (2, 2), padding='same', activation=tf.nn.relu, name='decode_3')(decoder_2)
        # print('Decoder 3: {}'.format(decoder_3))
        
        # output = [-1, 32, 32, 3]
        output = tf.layers.Conv2DTranspose(3, 3, strides=(1,1), padding='same', activation=tf.nn.relu, name='output')(decoder_3)
        # print('Output: {}'.format(output))
        
        return opt_metrics_autoencoder(x, code, output, lr, b_1, b_2)
        

def conv_block(inputs, filters, name):
    with tf.name_scope(name) as scope:
        # first hidden conv layer
        hidden_1 = tf.layers.Conv2D(filters=filters[0], kernel_size=3, padding='same', activation=tf.nn.relu, name=name+'_1')(inputs)
        
        # first pooling layer
        # pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        
        # second hidden conv layer
        hidden_2 = tf.layers.Conv2D(filters=filters[1], kernel_size=3, padding='same', activation=tf.nn.relu, name=name+'_2')(hidden_1)

        b_norm_1 = tf.layers.BatchNormalization(trainable=True)(hidden_2)

        pool = tf.layers.MaxPooling2D(2, 2, padding='same')

        output_tensor=pool(b_norm_1)

        layer_list = [hidden_1, hidden_2, b_norm_1, pool]

        block_parameter_num = sum(map(lambda layer: layer.count_params(), layer_list))
        print('Number of parameters in conv block with {} input: {}'.format(inputs.shape, block_parameter_num))
        return output_tensor

def sep_conv_block(inputs, filters, name, apply_pool):
    with tf.name_scope(name) as scope:
        # first hidden conv layer
        hidden_1 = tf.layers.SeparableConv2D(filters=filters[0], kernel_size=3, padding='same', 
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), activation=tf.nn.relu, name=name+'_1')(inputs)
        
        # first pooling layer
        # pool_1 = tf.layers.max_pooling2d(hidden_1, 2, 2, padding='same')
        
        # second hidden conv layer
        hidden_2 = tf.layers.SeparableConv2D(filters=filters[1], kernel_size=3, padding='same',
                                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1), activation=tf.nn.relu, name=name+'_2')(hidden_1)
        
        b_norm_1 = tf.layers.BatchNormalization(trainable=True)(hidden_2)                                      
        
        pool = tf.layers.MaxPooling2D(2, 2, padding='same')
        
        if apply_pool:
                output_tensor=pool(b_norm_1)
                layer_list = [hidden_1, hidden_2, b_norm_1, pool]

                block_parameter_num = sum(map(lambda layer: layer.count_params(), layer_list))
                print('Number of parameters in conv block with {} input: {}'.format(inputs.shape, block_parameter_num))

                return output_tensor

        else:
                layer_list = [hidden_1, hidden_2, b_norm_1]

                block_parameter_num = sum(map(lambda layer: layer.count_params(), layer_list))
                print('Number of parameters in conv block with {} input: {}'.format(inputs.shape, block_parameter_num))

                return b_norm_1

        
        

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

        conv_block_1 = sep_conv_block(x,[8, 16], name='conv1', False)

        dropout_1 = tf.layers.Dropout(rate=0.3, name='dropout_1')(conv_block_1)

        conv_block_2 = sep_conv_block(dropout_1, [32,64], name='conv2', True)

        dropout_2 = tf.layers.Dropout(rate=0.3, name='dropout_2')(conv_block_2)

        conv_block_3 = sep_conv_block(dropout_2,[128, 256], name='conv3', False)

        dropout_3 = tf.layers.Dropout(rate=0.3, name='dropout_1')(conv_block_3)

        conv_block_4 = sep_conv_block(dropout_3, [512,1024], name='conv4', True)

        dropout_4 = tf.layers.Dropout(rate=0.3, name='dropout_2')(conv_block_4)

        # flatten from 4D to 2D for dense layer
        flat = tf.reshape(dropout_4, [-1, 8*8*1024])

        # dense layer 1
        dense_1 = tf.layers.Dense(2048, name='dense_1', activation=tf.nn.relu)(flat)

        # dense layer 2
        dense_2 = tf.layers.Dense(1024, name='dense_2', activation=tf.nn.relu)(dense_1)

        # dense layer 2
        dense_3 = tf.layers.Dense(512, name='dense_3', activation=tf.nn.relu)(dense_2)

        # dense layer output
        output = tf.layers.Dense(100, name='output_layer')(dense_3)

        tf.identity(output, name='output')
        # get the model metrics
        confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs= opt_metrics(x, y, output, lr, b_1, b_2)
        
        return confusion_matrix, cross_entropy, train_op, global_step_tensor, saver, accuracy, lhs, rhs



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
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
        train_op = tf.group([train_op, update_ops])

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
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = optimizer.minimize(total_loss, global_step=global_step_tensor)
        train_op = tf.group([train_op, update_ops])

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
