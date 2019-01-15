# coding: utf-8
import tensorflow as tf
import numpy as np
'''
 generally , we use linear function instead of custom function
     define four parameters in function add_layer:
        -   inputs: the value to be input
        -   size: the size of the value
        -   out_size: the size of output value
        -   activation_function: the function of activation, we set it to use linear equations by default 
'''


def add_layer(inputs, in_size, out_size, activation_function=None):
    ## normal_distribution is better than zeros ##
    Layer_Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    ## donnot recommend zero ##
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    ## the unactivatived value of the neural netework ##
    Wx_plus_b = tf.matmul(inputs, Layer_Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

## train data, to simulate the data we need to train
x_data = np.linspace(-1, 1, 300)[:, np.newaxis] # 300line
noise = np.random.normal(0, 0.05, x_data.shape) # variance(mean) is 0.05 , type is the same to x_data
## the norm data
y_data = np.square(x_data) - 0.5+noise

'''
 the data trim to be input , tell the system : i am going to enter some data like this
    -  'tf.float32': the type of data we are going to input
    -   'None': it means whatever the number of input data is
    -   '1' : the number of features, here we only one
'''
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

'''
 input value : xs
    -  10 neurals
    -  use relu as activation_function
'''
layer1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(layer1, 10, 1, activation_function=None)

## the difference between prediction  and y_data  which is the norm(real) data ##
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                                    reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  ## learning rate. to minimize(reduce) the loss

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    ## now , start
    sess.run(init_op)
    for i in range(1000):

        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 200 == 0:
            ## to see the improvement
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
