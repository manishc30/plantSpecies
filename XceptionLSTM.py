#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 18:15:11 2019

@author: manish
"""
import datetime
from tqdm import tqdm
from keras.applications import xception
from sklearn.metrics import confusion_matrix
import sklearn
import tensorflow as tf
import seaborn as sns
from speciesUtilities import species, num_species, train_dir
from speciesUtilities import generate_images, augment_train_images
from speciesUtilities import createSegmentedImage, preprocess_image, denseEncodingToOne, oneEncodingToDense,readImage
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
from keras.engine import Input
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Lambda, K, LSTM
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import merge,concatenate
from keras.models import Model
from keras.callbacks import TensorBoard
from keras.optimizers import Adam
import time
import numpy as np
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report
# start timer
startTime = datetime.datetime.now();


# validation set size
validationSetSize = 10;

# kaggle
copyModelsToLocal = False # set True to download keras pretrained models

# train data
takeLimitedTrainData = False # set True to take only sample of train data for less running time
limitedTrainDataPerSpecies = 200 # number of train data per species if takeLimitedTrainData is true
loadFeaturesOfTrainData = True # set True to load features from npy file. If running first time, set it to True

# test data
takeLimitedTestData = False # set True to take only sample of test data for less running time
limitedTestData = 200 # number of test data samples if takeLimitedTestData is true
loadFeaturesOfTestData = True # set True to load features from npy file. If running first time, set it to True

# augmented images
useAugmentedData = False # set True to use augmented images
loadFeaturesOfAugData = True # set True to load features from file for augmented data. If running first time, set it to True

augmentSegmentedImages = False

plotImages = False
# Show the current directory
print('current directory:')
#!ls  
print('\nparent directory:')
#!ls ..  

# setting current directory
os.chdir('/Users/manish/Documents/XceptionSegmented/plant-seedlings-classification')

#Analyze data
print('input directory:')
#!ls /Users/manish/Documents/Seed/plant-seedlings-classification

print('\nfolders containing images of the corresponding species:')
#!ls /Users/manish/Documents/Seed/plant-seedlings-classification/train

if augmentSegmentedImages:
    print('Augmentation started')
    augStart = datetime.datetime.now()
    train_df, test_df = augment_train_images()
    print('Augmentation running time = ' ,datetime.datetime.now() - augStart)
    print('Plant images after augmentation')
    for sp in species:
        print('{} images of {}'.format(len(os.listdir(os.path.join(train_dir, sp))),sp))



    
## take a fixed number of samples for testing purpose
if takeLimitedTrainData:
    train_df = pd.concat([train_df[train_df['species'] == sp][:limitedTrainDataPerSpecies] for sp in species])
    train_df.index = np.arange(len(train_df))

if takeLimitedTestData:
    test_df = test_df[:limitedTestData]
    

print('Preprocessing training images')

targetImageSize = 299 # Bcoz Xception model takes this size only

# read, preprocess training and validation images  
x_train_valid = np.zeros((len(train_df), targetImageSize, targetImageSize, 3),
                         dtype='float32')
y_train_valid = train_df.loc[:, 'species_id'].values 
for i, filepath in tqdm(enumerate(train_df['filepath'])):

    img = readImage(filepath)
    # all pixel values are now between -1 and 1
    x_train_valid[i] = preprocess_image(np.expand_dims(img.copy().astype(np.float), axis=0)) 

print('Preprocessing test images')

# read, preprocess test images  
x_test = np.zeros((len(test_df), targetImageSize, targetImageSize, 3), dtype='float32')
for i, filepath in tqdm(enumerate(test_df['filepath'])):
    # read image
    img = readImage(filepath)
    # all pixel values are now between -1 and 1
    x_test[i] = preprocess_image(np.expand_dims(img.copy().astype(np.float), axis=0)) 
    
print('x_train_valid.shape = ', x_train_valid.shape)
print('x_test.shape = ', x_test.shape)

# show some examples
if plotImages:
    imgs = (((x_train_valid[0:4]+1.)/2.)*255.) # transform pixels into range [0,255]
    imgs_generated = imgs

    fig, axs = plt.subplots(4, 8, figsize=(20, 10))
    for i in range(8):
        axs[0,i].imshow(imgs_generated[0].astype(np.uint8))
        axs[1,i].imshow(imgs_generated[1].astype(np.uint8))
        axs[2,i].imshow(imgs_generated[2].astype(np.uint8))
        axs[3,i].imshow(imgs_generated[3].astype(np.uint8))   
        imgs_generated = generate_images(imgs)
        
## copy xception models into ~.keras directory

# set True when running on kaggle
if copyModelsToLocal:
    # create cache and models directory
    cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    models_dir = os.path.join(cache_dir, 'models')
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # show available pretrained keras models
    #!ls /Users/manish/Downloads/keras-pretrained-models/

    # copy xception models to models directory
    print('')
    print('use xception models')
    #!cp /Users/manish/Downloads/keras-pretrained-models/xception* ~/.keras/models/
    #!ls ~/.keras/models
  
def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)

def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]


input_tensor = Input(shape=(299, 299, 3))

## compute or load bottleneck features from xception model
def buildModel():
    img_width = 299
    img_height = 299

    print("Building model...")
    input_tensor = Input(shape=(img_width, img_height, 3))

    inc_path = '../input/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

    # Creating CNN
    #cnn_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')

    cnn_model = xception.Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)

    x = cnn_model.output

    cnn_bottleneck = GlobalAveragePooling2D()(x)
    #cnn_bottleneck = cnn_model.predict(x_train_valid, batch_size=32, verbose=1)
    
    # Make CNN layers not trainable
    for layer in cnn_model.layers:
        layer.trainable = False
    
    # Creating RNN
    x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(input_tensor)
    x = Reshape((23, 3887))(x)  # 23 timesteps, input dim of each timestep 3887
    x = LSTM(2048, return_sequences=True)(x)
    rnn_output = LSTM(2048)(x)

    # Merging both cnn bottleneck and rnn's output wise element wise multiplication
    x = concatenate([cnn_bottleneck, rnn_output])
    #predictions = Dense(12, activation='softmax')(x)

    model = Model(input=input_tensor, output=x)

    print("Model built")

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


# train and validation images
if not loadFeaturesOfTrainData: # if set to False
    
    print('x_train_valid.shape = ', x_train_valid.shape)
    print('y_train_valid.shape = ', y_train_valid.shape)
    print('')

    print('Training data feature extraction from Xception started')

    local_start = datetime.datetime.now()
    
    # load xception base model and predict the last layer comprising 2048 neurons per image
    base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
    x_train_valid_bf = base_model.predict(x_train_valid, batch_size=32, verbose=1)

    print('Feature extraction running time for train data: ', datetime.datetime.now()-local_start)    
    print('')
    print('x_train_valid_bf.shape = ', x_train_valid_bf.shape)

    print('')
    print('save bottleneck features and labels for later ')
    np.save(os.path.join(os.getcwd(),'x_train_valid_bf.npy'), x_train_valid_bf)
    np.save(os.path.join(os.getcwd(),'y_train_valid.npy'), y_train_valid)

else:
    # load bottleneck features and labels
    
    print('loading features from npy file')
    
    x_train_valid_bf = np.load(os.path.join(os.getcwd(),'x_train_valid_bf.npy'))
    y_train_valid = np.load(os.path.join(os.getcwd(),'y_train_valid.npy'))

    print('x_train_valid_bf.shape = ', x_train_valid_bf.shape)
    print('y_train_valid.shape = ', y_train_valid.shape)
    
# test images
if not loadFeaturesOfTestData: # if set to False
    # compute bottleneck features from xception model
    print('Test data feature extraction from Xception started')
    local_start = datetime.datetime.now()
    
    # load xception base model and predict the last layer comprising 2048 neurons per image
    base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
    x_test_bf = base_model.predict(x_test, batch_size=32, verbose=1)
    
    print('Features extraction running time for test data: ', datetime.datetime.now()-local_start)    
    print('x_test_bf = ',x_test_bf.shape)

    print('saving features in x_test_bf.npy file')
    np.save(os.path.join(os.getcwd(),'x_test_bf.npy'), x_test_bf)

else:
    # load bottleneck features and compute the predictions
    print('loading features from x_test_bf_of_segmented_images.npy file')
    x_test_bf = np.load(os.path.join(os.getcwd(),'x_test_bf.npy'))
    print('x_test_bf.shape = ', x_test_bf.shape)
    

## compute or load bottleneck features for augmented data

if useAugmentedData:

    if not loadFeaturesOfAugData:  # if set to False

        for i in range(2):
            x_aug_tmp = generate_images(x_train_valid)
            y_aug_tmp = y_train_valid

            print('compute bottleneck features from Xception network')
            print('x_aug_tmp.shape = ', x_aug_tmp)
            print('y_aug_tmp.shape = ', y_aug_tmp)

            local_start = datetime.datetime.now()

            # load xception model and predict the last layer having 2048 neurons per image
            base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
            x_aug_tmp_bf = base_model.predict(x_aug_tmp, batch_size=32, verbose=1)

            print('running time: ', datetime.datetime.now()-local_start)    
            print('')
            print('x_aug_tmp_bf.shape = ', x_aug_tmp_bf.shape)
            
            if i==0:
                x_aug = x_aug_tmp_bf
                y_aug = y_aug_tmp
            else:
                x_aug = np.concatenate([x_aug,x_aug_tmp_bf])
                y_aug = np.concatenate([y_aug,y_aug_tmp])
            
            print('')
            print('save bottleneck features and labels for later ')
            np.save(os.path.join(os.getcwd(),'x_aug.npy'), x_aug)
            np.save(os.path.join(os.getcwd(),'y_aug.npy'), y_aug)

     
    else:
        # load bottleneck features and compute the predictions

        print('load bottleneck features')
        x_aug_bf = np.load(os.path.join(os.getcwd(), 'x_aug_bf_of_segmented_images.npy'))
        y_aug = np.load(os.path.join(os.getcwd(), 'y_aug_of_segmented_images.npy'))

        print('x_aug_bf.shape = ', x_aug_bf.shape)
        print('y_aug.shape = ', y_aug.shape)
        
##############################################################[16]
## combine files
if False:
    x_aug_bf_1 = np.load(os.path.join(os.getcwd(), 'x_aug_bf_of_segmented_images.npy'))
    y_aug_1 = np.load(os.path.join(os.getcwd(), 'y_aug_of_segmented_images.npy'))
    print(x_aug_bf_1.shape, y_aug_1.shape)
    x_aug_bf_2 = np.load(os.path.join(os.getcwd(), 'x_aug_bf.npy'))
    y_aug_2 = np.load(os.path.join(os.getcwd(), 'y_aug.npy'))
    print(x_aug_bf_2.shape, y_aug_2.shape)
    x_aug_bf = np.concatenate([x_aug_bf_1,x_aug_bf_2])
    y_aug = np.concatenate([y_aug_1,y_aug_2])
    print(x_aug_bf.shape,y_aug.shape)


if validationSetSize > 0:
    # split into train and validation sets
    valid_set_size = int(len(x_train_valid_bf) * validationSetSize/100);
    trainDataSize = len(x_train_valid_bf) - valid_set_size;
else:
    # train on all available data
    valid_set_size = int(len(x_train_valid_bf) * 0.1);
    trainDataSize = len(x_train_valid_bf)

def shuffle_train_valid_data():
    
    print('shuffle train and validation data')
    
    # shuffle train and validation data of original data
    perm_array = np.arange(len(x_train_valid_bf)) 
    np.random.shuffle(perm_array)
    
    # split train and validation sets based on original data
    x_train_feature = x_train_valid_bf[perm_array[:trainDataSize]]
    y_train = denseEncodingToOne(y_train_valid[perm_array[:trainDataSize]], num_species)
    x_valid_feature = x_train_valid_bf[perm_array[-valid_set_size:]]
    y_valid = denseEncodingToOne(y_train_valid[perm_array[-valid_set_size:]], num_species)
    
    # augment train data by generated images
    if useAugmentedData:
        
        x_train_feature = np.concatenate([x_train_feature, x_aug_bf])
        y_train = np.concatenate([y_train, denseEncodingToOne(y_aug, num_species)])
        
        # shuffle data
        perm_array = np.arange(len(x_train_feature)) 
        np.random.shuffle(perm_array)
        
        x_train_feature = x_train_feature[perm_array]
        y_train = y_train[perm_array]
         
    return x_train_feature, y_train, x_valid_feature, y_valid 

# split into train and validation sets including shuffling
x_train_feature, y_train, x_valid_feature, y_valid = shuffle_train_valid_data() 

print('x_train_feature.shape = ', x_train_feature.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid_feature.shape = ', x_valid_feature.shape)
print('y_valid.shape = ', y_valid.shape)


## neural network with tensorflow

# permutation array for shuffling train data
perm_array_train = np.arange(len(x_train_feature)) 
index_in_epoch = 0

def getBatch(batch_size):
    
    global index_in_epoch, perm_array_train
  
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > trainDataSize:
        np.random.shuffle(perm_array_train) # shuffle data
        start = 0 # start next epoch
        index_in_epoch = batch_size
              
    end = index_in_epoch
    
    return x_train_feature[perm_array_train[start:end]], y_train[perm_array_train[start:end]]

x_size = x_train_feature.shape[1] 
y_size = num_species 
n_n_fc1 = 1024 
n_n_fc2 = num_species # number of neurons

# variables for input and output 
x_data = tf.placeholder('float', shape=[None, x_size])
y_data = tf.placeholder('float', shape=[None, y_size])

# 1.layer: fully connected
W_fc1 = tf.Variable(tf.truncated_normal(shape = [x_size, n_n_fc1], stddev = 0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [n_n_fc1]))  
h_fc1 = tf.nn.relu(tf.matmul(x_data, W_fc1) + b_fc1)

# add dropout
tf_keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, tf_keep_prob)

# 3.layer: fully connected
W_fc2 = tf.Variable(tf.truncated_normal(shape = [n_n_fc1, n_n_fc2], stddev = 0.1)) 
b_fc2 = tf.Variable(tf.constant(0.1, shape = [n_n_fc2]))  
z_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, 
                                                                       logits=z_pred));

# optimisation function
learningRate = tf.placeholder(dtype='float', name="learningRate")
train_step = tf.train.AdamOptimizer(learningRate).minimize(cross_entropy)

# evaluation
y_pred = tf.cast(tf.nn.softmax(z_pred), dtype = tf.float32);
y_pred_class = tf.cast(tf.argmax(y_pred,1), tf.int32)
y_data_class = tf.cast(tf.argmax(y_data,1), tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, y_data_class), tf.float32))

# parameters
crossValidations = 1 # number of cross validations
n_epoch = 15#15 # number of epochs. n_epoch, batch_size and keep_prob changes the accuracy of the model
batch_size = 50 #50 
keep_prob = 0.33#0.33 # dropout
learn_rate_range = [0.01,0.005,0.0025,0.001,0.001,0.001,0.00075,0.0005,0.00025,0.0001,
                   0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001];
learn_rate_step = 3 # in terms of epochs

trainingAccuracy = 0
validationAccuracy = 0
trainingLoss = 0
validationLoss = 0
y_test_predicted_prob = 0
y_validation_pred = 0

train_loss_list = []
train_acc_list = []
valid_loss_list = []
valid_acc_list = []
epochs_list = []
# use cross validation
for j in range(crossValidations):
    
    # start TensorFlow session and initialize global variables
    sess = tf.InteractiveSession() 
    sess.run(tf.global_variables_initializer())  

    # shuffle train/validation splits
    shuffle_train_valid_data() 
    n_step = -1;
    
    # training model
    for i in range(int(n_epoch*trainDataSize/batch_size)):

        if i%int(learn_rate_step*trainDataSize/batch_size) == 0:
            n_step += 1;
            learn_rate = learn_rate_range[n_step];
            print('learnrate = ', learn_rate)
        
        x_batch, y_batch = getBatch(batch_size)
        
        sess.run(train_step, feed_dict={x_data: x_batch, y_data: y_batch, 
                                        tf_keep_prob: keep_prob, 
                                        learningRate: learn_rate})

        if i%int(0.25*trainDataSize/batch_size) == 0:
            
            train_loss = sess.run(cross_entropy,
                                  feed_dict={x_data: x_train_feature[:valid_set_size], 
                                             y_data: y_train[:valid_set_size], 
                                             tf_keep_prob: 1.0})

            
            train_acc = accuracy.eval(feed_dict={x_data: x_train_feature[:valid_set_size], 
                                                 y_data: y_train[:valid_set_size], 
                                                 tf_keep_prob: 1.0})    

            valid_loss = sess.run(cross_entropy,feed_dict={x_data: x_valid_feature, 
                                                           y_data: y_valid, 
                                                           tf_keep_prob: 1.0})

           
            valid_acc = accuracy.eval(feed_dict={x_data: x_valid_feature, 
                                                 y_data: y_valid, 
                                                 tf_keep_prob: 1.0})      
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            valid_loss_list.append(valid_loss)
            valid_acc_list.append(valid_acc)
            epochs_list.append((i+1)*batch_size/trainDataSize)
            print('%.2f epoch: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(
                (i+1)*batch_size/trainDataSize, train_loss, valid_loss, train_acc, valid_acc))

    
    trainingAccuracy += train_acc
    validationAccuracy += valid_acc
    trainingLoss += train_loss
    validationLoss += valid_loss
    
    y_validation_pred += y_pred.eval(feed_dict={x_data: x_valid_feature, tf_keep_prob: 1.0}) 
    y_test_predicted_prob += y_pred.eval(feed_dict={x_data: x_test_bf, tf_keep_prob: 1.0}) 

    sess.close()
        
trainingAccuracy /= float(crossValidations)
validationAccuracy /= float(crossValidations)
trainingLoss /= float(crossValidations)
validationLoss /= float(crossValidations)

# plot the model loss and accuracy

x = [(i+1) for i in range(len(epochs_list))]

f,ax = plt.subplots(1,2, figsize=(12,5))
ax[0].plot(epochs_list, train_loss_list)
ax[0].plot(epochs_list, valid_loss_list)
ax[0].set_title("Loss plot")
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("loss")
ax[0].legend(['train', 'valid'])


ax[1].plot(x, train_acc_list)
ax[1].plot(x, valid_acc_list)
ax[1].set_title("Accuracy plot")
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("acc")
ax[1].legend(['train', 'valid'])
plt.show()

# Deleting the lists after plotting
del train_loss_list
del valid_loss_list
del train_acc_list
del valid_acc_list

# final validation prediction
y_validation_pred /= float(crossValidations)
y_valid_pred_class = np.argmax(y_validation_pred, axis = 1) # Range is from 0 to 11

# final test prediction
y_test_predicted_prob /= float(crossValidations)
y_test_pred_class_DNN = np.argmax(y_test_predicted_prob, axis = 1) # Range is from 0 to 11

# final loss and accuracy
print('final: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(trainingLoss, 
                                                                      validationLoss, 
                                                                      trainingAccuracy, 
                                                                      validationAccuracy))


print(sklearn.metrics.classification_report(oneEncodingToDense(y_valid),y_valid_pred_class))
print('Average Precision: ',sklearn.metrics.precision_score(oneEncodingToDense(y_valid),y_valid_pred_class,average='weighted'))
print('Average Recall: ',sklearn.metrics.recall_score(oneEncodingToDense(y_valid),y_valid_pred_class,average='weighted'))

################################################################
## show confusion matrix


if plotImages:
    
    cnf_matrix = confusion_matrix(oneEncodingToDense(y_valid), y_valid_pred_class)

    abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
    pd.DataFrame({'class': species, 'abbreviation': abbreviation})

    fig, ax = plt.subplots(1)
    ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True,fmt="d")
    ax.set_xticklabels(abbreviation)
    ax.set_yticklabels(abbreviation)
    plt.title('Confusion matrix of validation set')
    plt.ylabel('True species')
    plt.xlabel('Predicted species')
    plt.show();


##########################################################[24]
print('total running time: ', datetime.datetime.now()-startTime)