from scipy import io as sio
import numpy as np
import cv2
import tensorflow as tf
import os
import random

from CNN_Structure import CNNStructure

tf.reset_default_graph()

def inverte(imagem):
    new_im = (255-imagem)
    return new_im
    
emnist = sio.loadmat('matlab/emnist-byclass.mat')

#load training
x_train = emnist["dataset"][0][0][0][0][0][0] 
x_train = x_train.astype(np.float32)
y_train = emnist["dataset"][0][0][0][0][0][1] #labels

# load test
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)
y_test = emnist["dataset"][0][0][1][0][0][1]

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1, order="F")
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1, order="F")

train_labels = y_train
test_labels = y_test

class_idx = ['0','1','2','3','4','5','6','7','8','9',
             'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
             'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

print("Starting count")
counter=0
for i in range(len(x_train)): 
    index = train_labels[i][0]
    if(index <= 35): #select only capital letters and numbers
        counter += 1
print("Finished count")
new_x_train = np.zeros((counter, 28, 28, 1), dtype=np.float32) #counter = #labels
new_train_labels = np.zeros((counter, 1), dtype=np.uint8)
print("Starting generation of new train")
j=0
for i in range(len(x_train)): #fill a vector with just the desired images
    index = train_labels[i][0]
    if(index <= 35):
        new_x_train[j,:,:,:]=inverte(x_train[i,:,:,:])
        new_train_labels[j,:]=train_labels[i,:]
        j = j+1

number_classes=36
new_y_train=np.eye(number_classes)[new_train_labels]
new_y_train=np.squeeze(new_y_train)
print("Finished")

#same for test set
print("Starting count")
counter=0
for i in range(len(x_test)):
    index = test_labels[i][0]
    if(index <= 35):
        counter += 1
print("Finished count")
new_x_test = np.zeros((counter, 28, 28, 1), dtype=np.float32)
new_test_labels = np.zeros((counter, 1), dtype=np.uint8)
print("Starting generation of new train")
j=0
for i in range(len(x_test)):
    index = test_labels[i][0]
    if(index <= 35):
        new_x_test[j,:,:,:]=inverte(x_test[i,:,:,:])
        new_test_labels[j,:]=test_labels[i,:]
        j = j+1

number_classes=36
new_y_test=np.eye(number_classes)[new_test_labels]
new_y_test=np.squeeze(new_y_test)
print("Finished")

print("Start neural network building")
dirname = os.path.dirname(__file__)
epochs_to_read = 1
num_examples_train = 64
batch_size_training = num_examples_train/4
num_examples_val = 32
batch_size_validation = num_examples_val/4
reg_constant = 1e-5
learning_rate = 1e-3
filter_depth = 8
shape=(28,28)
num_input_channels = 1

# Training Parameters
training_set_size = len(new_x_train)
learning_rate = 0.0001
reg_constant = 1e-5
training_batch_size = 128
steps_per_epoch = int(training_set_size/training_batch_size)
num_epoch = 10
num_steps = num_epoch*steps_per_epoch

# Network Parameters
num_classes = 36 # MNIST total classes (0-9 digits, upper case letters)

model = CNNStructure(shape, num_input_channels, num_classes)

x_image, y_, layer_conv1, weights_conv1, layer_conv2, weights_conv2, flatten, num_filters_flatten, fc1, fc1_out, y_conv, fc_read_out = model.getStructure()

reg_losses = tf.reduce_sum([tf.nn.l2_loss(weights_conv1), tf.nn.l2_loss(weights_conv2), tf.nn.l2_loss(fc1)])

print("Import Structure ok")

# Construct model
prediction = tf.nn.softmax(y_conv)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op + reg_constant*(reg_losses))


# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


print("Start training")
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

saver = tf.train.Saver()

#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True

f = open(dirname + "/accuracy.txt","w+")

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)
    start_index=0 #start from epoch 0
    epoch=0
    for step in range(1, num_steps+1):
        end_index=start_index+training_batch_size 
        if(end_index>training_set_size):
            end_index = training_set_size - 1 #to avoid array index out of bounds

        batch_x = new_x_train[start_index:end_index,:,:,:]
        batch_y = new_y_train[start_index:end_index,:]
        
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={x_image: batch_x, y_: batch_y})
        if step % steps_per_epoch == 0 or step == 1: #if an epoch is done
            # Calculate batch loss and accuracy
            loss_t, acc_t = sess.run([loss_op, accuracy], feed_dict={x_image: batch_x, y_: batch_y}) #get loss and acc with such input (x_image, y_)
            
            print("Epoch " + str(epoch) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss_t) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc_t))
            
            strt = random.randint(0,len(new_x_test)-500) #get a random index
            batch_x_val = new_x_test[strt:strt+500,:,:,:] #get 500 images from test set
            batch_y_val = new_y_test[strt:strt+500,:]
            acc_v = sess.run(accuracy, feed_dict={x_image: batch_x_val, y_: batch_y_val}) #get acuracy
            
            print("Epoch " + str(epoch) + ", Validation Accuracy= " + \
                  "{:.3f}".format(acc_v))
            f.write(str(acc_t) + "\t" + str(acc_v) + "\n")
            
            epoch += 1
            
        start_index = (start_index+training_batch_size)%training_set_size #reset start index
    print("Optimization Finished!")
    saver.save(sess, dirname + '/new_session_char.ckpt') #save session in a file (weights of the CNN)
    