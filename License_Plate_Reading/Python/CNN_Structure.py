import tensorflow as tf
#import numpy as np


def weight_variable(shape):
#  initializer = tf.contrib.layers.xavier_initializer_conv2d(uniform=False)
  initializer = tf.contrib.layers.xavier_initializer(uniform=False)
  #initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initializer(shape=shape), name='weight')

def bias_variable(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='VALID')
  
def max_pool_3x3(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 3, 3, 1], padding='VALID')

def max_pool_4x4(x):
  return tf.nn.max_pool(x, ksize=[1, 4, 4, 1],
                        strides=[1, 4, 4, 1], padding='VALID')  

def pad_symmetric(input_data, kernel_size):
    paddings = tf.cast([[0, 0],[(kernel_size-1)/2, (kernel_size-1)/2],
                        [(kernel_size-1)/2, (kernel_size-1)/2], [0, 0]], tf.int32)
    return tf.pad(input_data, paddings, "SYMMETRIC")


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True,
                   pooling=1):           # Use max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = weight_variable(shape=shape)

    # Create new biases, one for each filter.
    biases = bias_variable(length=num_filters)
    
    input = pad_symmetric(input, filter_size)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='VALID')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        if(pooling==2):
            layer = max_pool_2x2(layer)
        elif pooling==3:
            layer = max_pool_3x3(layer)
        elif(pooling==4):
            layer = max_pool_4x4(layer)    
        else:
            layer=layer


    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


def new_deconv_layer(input,              # The previous layer.
                     num_input_channels, # Num. channels in prev. layer.
                     filter_size,        # Width and height of each filter.
                     num_filters,        # Number of filters.
                     stride_shape): 
      
#    with tf.variable_scope(layer_name):
        print('Input shape = ', input.get_shape().as_list())
    #    print('Batch size = ', tf.shape(input).get_shape()[0])
    #    input_channel_size = input.get_shape().as_list()[3]
        input_size_h = input.get_shape().as_list()[1]
        input_size_w = input.get_shape().as_list()[2]
    #    stride_shape = [1, stride_h, stride_w, 1]
        output_size_h = (input_size_h)*stride_shape[1]
        output_size_w = (input_size_w)*stride_shape[2]
        output_shape = tf.stack([tf.shape(input).get_shape()[0], output_size_h, output_size_w, num_filters])
        print('Output shape = ', output_shape)
    #creating weights:
        filter_shape = [filter_size, filter_size, num_filters, num_input_channels]
        weights = weight_variable(shape=filter_shape)
          
        biases = bias_variable(length=num_filters)
          
#        layer = tf.nn.conv2d_transpose(input, weights, output_shape, stride_shape, padding='SAME')
        layer = tf.layers.conv2d_transpose(inputs=input, filters=num_filters, kernel_size=filter_size, strides=(stride_shape[1],stride_shape[2]), padding="SAME")
        layer += biases
        
        layer = tf.nn.relu(layer)
        
        print("New output shape = ", layer.get_shape())
          
          #Now output.get_shape() is equal (?,?,?,?) which can become a problem in the 
          #next layers. This can be repaired by reshaping the tensor to its shape:
    #    layer = tf.reshape(layer, output_shape)
          #now the shape is back to (?, H, W, C) or (?, C, H, W)
          
        return layer, weights

  
def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()

    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = weight_variable(shape=[num_inputs, num_outputs])
    biases = bias_variable(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer, weights

def batch_norm(x, n_out, phase_train):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed


class CNNStructure(object):
    
    #--------------------------------------------------------------------------
    #-------------------------------CNN Structure------------------------------
    #--------------------------------------------------------------------------
    
    def __init__ (self, shape, num_input_channels, num_classes):
        
        self._shape=shape
        
        self._x_image = tf.placeholder(tf.float32, shape=[None, shape[1], shape[0], num_input_channels])#shape=[None, 128, 128, 6])
        self._y_ = tf.placeholder(tf.float32, shape=[None, 36])#[None, 128, 128])        
        
        #first layer
        filter_size = 5
        num_filters1 = 32
        self._layer_conv1, self._weights_conv1 = new_conv_layer(input=self._x_image, num_input_channels=num_input_channels, filter_size=filter_size, num_filters=num_filters1, use_pooling=True, pooling=2)
        
        #second layer
        filter_size2 = 5
        num_filters2 = 64
        self._layer_conv2, self._weights_conv2 = new_conv_layer(input=self._layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True, pooling=3)
        
        #flatten layer
        self._flatten, self._num_filters_flatten = flatten_layer(self._layer_conv2)
        
        #fully connected layer
        num_out_fc1=1024
        self._fc1, self._fc1_out = new_fc_layer(input=self._flatten, num_inputs=self._num_filters_flatten, num_outputs=num_out_fc1, use_relu=True)
        
        #read out layer
        self._y_conv, self._fc_read_out = new_fc_layer(input=self._fc1, num_inputs=num_out_fc1, num_outputs=num_classes, use_relu=True)
        
    def getStructure(self):
#        return self._x_image, self._y_, self._layer_conv1, self._weights_conv1, self._layer_conv2, self._weights_conv2, self._layer_conv3, self._weights_conv3, self._layer_conv4, self._weights_conv4, self._y_conv_pre, self._weights_conv5, self._y_conv
        return self._x_image, self._y_, self._layer_conv1, self._weights_conv1, self._layer_conv2, self._weights_conv2, self._flatten, self._num_filters_flatten, self._fc1, self._fc1_out, self._y_conv, self._fc_read_out
