from scipy import io as sio
import numpy as np
import cv2
import tensorflow as tf
import os
import glob
#from dataset_reader_char import DataSet
from CNN_Structure import CNNStructure

tf.reset_default_graph() #reset default graph

class_idx = ['0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

print("Start neural network building")
dirname = os.path.dirname(__file__)
reg_constant = 1e-5 #to avoid overfitting
learning_rate = 1e-3
filter_depth = 8
shape=(28,28)
num_input_channels = 1

# Training Parameters
training_set_size = 533933
learning_rate = 0.0001
reg_constant = 1e-5
training_batch_size = 16
steps_per_epoch = int(training_set_size/training_batch_size)
num_epoch = 20
num_steps = num_epoch*steps_per_epoch

# Network Parameters
num_classes = 36 # MNIST total classes (0-9 digits, upper case letters)

model = CNNStructure(shape, num_input_channels, num_classes)

x_image, y_, layer_conv1, weights_conv1, layer_conv2, weights_conv2, flatten, num_filters_flatten, fc1, fc1_out, y_conv, fc_read_out = model.getStructure()

reg_losses = tf.reduce_sum([tf.nn.l2_loss(weights_conv1), tf.nn.l2_loss(weights_conv2), tf.nn.l2_loss(fc1)]) #to avoid overfitting

print("Import Structure ok")

# Construct model
prediction = tf.nn.softmax(y_conv) #to normalize the output sum to 1 

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op + reg_constant*(reg_losses))


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1)) #get the correct output
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32)) #get the mean of correct predictions


print("Start classification")
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer() #initialize environment variables in tensorflow

saver = tf.train.Saver() #to further import weights of CNN

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True

# Start training
with tf.Session() as session:

    # Run the initializer
    session.run(init) #inside this session initialize environment variables
    saver.restore(session, dirname + '/cnn_storage/new_session_char.ckpt') #import weights
    extracted_images = glob.glob(dirname + '/chars/*.jpg')
    f = open(dirname + "/classified_chars.txt","w+")
    for i in range(len(extracted_images)):
        in_batch = cv2.imread(extracted_images[i])[:,:,0]
        batch_x = np.zeros((1,28,28,1),dtype=np.uint8)+255 #create a 28x28 array of a white image
        batch_x[0,4:24,4:24,0]=in_batch #to select the image without the white board
        tmp=session.run(prediction, feed_dict={x_image: batch_x}) #classify my input
        index=np.where(tmp==np.max(tmp)) #localize the index of the max value of prediction in the vector
        f.write(class_idx[index[1][0]]) #
    f.close()
