#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:22:08 2019

@author: manish
"""
import os
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from numpy import expand_dims

species = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
           'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
           'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']

num_species = len(species)

data_dir = '/Users/manish/Documents/XceptionSegmented/plant-seedlings-classification/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# Function to augment the train images to control the class imbalance
def augment_train_images():
    
    black_count = 0
    charlock_count = 0
    cleavers_count = 0
    chickwood_count = 0
    wheat_count = 0
    hen_count = 0
    loose_count = 0
    maize_count = 0
    mayweed_count = 0
    purse_count = 0
    cranesbill_count = 0
    beet_count = 0
    
    for sp in species:
        print('{} plant images of {}'.format(len(os.listdir(os.path.join(train_dir, sp))),sp))
        if(sp == species[0]):
            black_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[1]):
            charlock_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[2]):
            cleavers_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[3]):
            chickwood_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[4]):
            wheat_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[5]):
            hen_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[6]):
            loose_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[7]):
            maize_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[8]):
            mayweed_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[9]):
            purse_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[10]):
            cranesbill_count = len(os.listdir(os.path.join(train_dir, sp)))
        elif(sp == species[11]):
            beet_count = len(os.listdir(os.path.join(train_dir, sp)))
    
    
    #Calculating the number of times image augmentation has to be done for each species
    #Loose Silky-bent is augmented twice. All other species will be augmented based on Loose Silky-bent because it has the highest count
    loose_count = loose_count * 2

    diff_black=loose_count-black_count
    diff_charlock=loose_count- charlock_count
    diff_cleavers=loose_count - cleavers_count
    diff_chickwood = loose_count - chickwood_count
    diff_wheat = loose_count - wheat_count
    diff_hen = loose_count - hen_count
    diff_maize = loose_count - maize_count
    diff_mayweed = loose_count - mayweed_count
    diff_purse = loose_count - purse_count
    diff_cranesbill = loose_count - cranesbill_count
    diff_beet = loose_count - beet_count


    #Calculating the number of iterations required for image augmentation for each species
    black_itr = round(diff_black/black_count)
    charlock_itr = round(diff_charlock/charlock_count)
    cleavers_itr = round(diff_cleavers/cleavers_count)
    chickwood_itr = round(diff_chickwood/chickwood_count)
    wheat_itr = round(diff_wheat/wheat_count)
    hen_itr = round(diff_hen/hen_count)
    maize_itr = round(diff_maize/maize_count)
    mayweed_itr = round(diff_mayweed/mayweed_count)
    purse_itr = round(diff_purse/purse_count)
    cranesbill_itr = round(diff_cranesbill/cranesbill_count)
    beet_itr = round(diff_beet/beet_count)

    for i in species:
        path = train_dir + '/' + i
        images = [f for f in os.listdir(path) if os.path.splitext(f)[-1] == '.png']   
        
        if(i == species[6]):
            itr = 1
        elif(i == species[0]):
            itr = black_itr
        elif(i == species[1]):
            itr = charlock_itr
        elif(i == species[2]):
            itr = cleavers_itr
        elif(i == species[3]):
            itr = chickwood_itr
        elif(i == species[4]):
            itr = wheat_itr
        elif(i == species[5]):
            itr = hen_itr
        elif(i == species[7]):
            itr = maize_itr
        elif(i == species[8]):
            itr = mayweed_itr
        elif(i == species[9]):
            itr = purse_itr
        elif(i == species[10]):
            itr = cranesbill_itr
        elif(i == species[11]):
            itr = beet_itr
    

        for image in images:
            img = load_img(path + "/" + image)  
            data = img_to_array(img)
            samples = expand_dims(data, 0)
            datagen = ImageDataGenerator(rotation_range = 15, width_shift_range = 0.1 , height_shift_range = 0.1, horizontal_flip = False, vertical_flip = False, zoom_range = 0.1)
            it = datagen.flow(samples, batch_size=32, save_to_dir=path, save_prefix='AUG_PLANT_GEN_', save_format='png')
            for j in range(itr):
                batch = it.next()
    print('Updating the train_df and test_df after image augmentation')
    train_df, test_df = readTrainTestData()
    return train_df, test_df

def readTrainTestData():
    
    train = []
    for species_id, sp in enumerate(species):
        for file in os.listdir(os.path.join(train_dir, sp)):
            train.append(['train/{}/{}'.format(sp, file), file, species_id, sp])
    train_df = pd.DataFrame(train, columns=['filepath', 'file', 'species_id', 'species'])
    print('')
    print('train_df.shape = ', train_df.shape)

    # read all test data
    test = []
    for file in os.listdir(test_dir):
        test.append(['test/{}'.format(file), file])
    test_df = pd.DataFrame(test, columns=['filepath', 'file'])
    print('test_df.shape = ', test_df.shape)
    return train_df, test_df
    
# method to read an image and resize it
def readAndResizeimage(fileName, target_size=None):
    img = cv2.imread(os.path.join(data_dir, fileName), cv2.IMREAD_COLOR)
    img = cv2.resize(img.copy(), target_size, interpolation = cv2.INTER_AREA)
    return img

# method to read an image 
def readImage(fileName):
    img = cv2.imread(os.path.join(data_dir, fileName), cv2.IMREAD_COLOR)
    return img


# Method to create the segmented image after contour extraction
def createSegmentedImage(fileName, imgSize):
    img = cv2.imread(os.path.join(data_dir, fileName), cv2.IMREAD_COLOR)
    imgResized = cv2.resize(img.copy(), imgSize, interpolation = cv2.INTER_AREA)
    # Masking started
    image_hsv = cv2.cvtColor(imgResized, cv2.COLOR_BGR2HSV) # convert to HSV
    lower_hsv = np.array([25, 100, 50]) # lower range of HSV [60 - sensitivity, 100, 50]
    upper_hsv = np.array([95, 255, 255]) # Upper range [60 + sensitivity, 255, 255]
    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
    segmentedImage = cv2.bitwise_and(imgResized, imgResized, mask = mask)
    image_blurred = cv2.GaussianBlur(segmentedImage, (0, 0), 3) #segmentedImage, (0, 0), 3
    image_sharp = cv2.addWeighted(segmentedImage, 1.5, image_blurred, -0.5, 0)

    return imgResized, image_sharp

def saveSegmentedImages():
    segmented_dir = data_dir + 'Segmented'
    os.mkdir(segmented_dir)
    for sp in species:
        dest_dir = segmented_dir + '/' +sp + '/'
        os.mkdir(dest_dir)
        for i in os.listdir(os.path.join(train_dir, sp)):
            img, image_sharpen = createSegmentedImage(
                    sp+'/'+i,(299,299))
        
            fileName = i
            print('File name is ', fileName)
            cv2.imwrite(dest_dir + fileName, image_sharpen.astype(np.uint8))

# Method to preprocess the images to shape (299,299,3) and all values in the range [-1,1]
def preprocess_image(img):
    img /= 255.
    img -= 0.5
    img *= 2
    return img

# Method to generate new images by rotation for online augmentation
def generate_images(imgs):
    imgs_len = len(imgs)
    image_generator = ImageDataGenerator(
        rotation_range = 15, width_shift_range = 0.1 , height_shift_range = 0.1,
        horizontal_flip = False, vertical_flip = False, zoom_range = 0.1)
    imgs = image_generator.flow(imgs.copy(), np.zeros(imgs_len), batch_size=imgs_len, shuffle = False).next()    
    return imgs[0]


# Method to encode labels to one hot
def denseEncodingToOne(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

# Method to encode labels back to dense for confusion matrix
def oneEncodingToDense(labels_one_hot):
    labels_dense = np.where(labels_one_hot == 1)[1]      
    return labels_dense

