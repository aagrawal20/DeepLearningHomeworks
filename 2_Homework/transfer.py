import tensorflow as tf 

# get session
session = tf.Session()
# import graph
saver = tf.train.import_meta_graph(cifar_dir + 'homework_2.meta')
# restore model
saver.restore(session, cifar_dir + 'homework_2')
# print operations
print(session.graph.get_operations())

#graph
graph = session.graph
# tensors
x = graph.get_tensor_by_name('input_placeholder:0')
# code layer
code_layer = graph.get_tensor_by_name('conv_net_2d_1/Relu:0')
# code without gradients
code_no_grad = tf.reshape(tf.stop_gradient(code_layer), [-1, height, width, channels])
# dense 1
dense_1 = tf.layers.dense(code_no_grad, 1024, name="dense1")
# dense 2
dense_2 = tf.layers.dense(dense1, 512, name="dense2")
# output
output = tf.layers.dense(dense2, 100, name="output")