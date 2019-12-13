#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 22:44:59 2019

@author: manish
"""

from speciesUtilities import species, train_dir, num_species,saveSegmentedImages
from speciesUtilities import readImage, readAndResizeimage, createSegmentedImage, readTrainTestData
import matplotlib.pyplot as plt
import mpl_toolkits.axes_grid1
import numpy as np
import os

plotImages = True # set False to reduce notebook running time
calculate_image_size = False
createSegmentedImages = False
# print number of images of each species in the training data
for sp in species:
    print('{} images of {}'.format(len(os.listdir(os.path.join(train_dir, sp))),sp))

if createSegmentedImages:
    saveSegmentedImages()
    

# read all train data

train_df, test_df = readTrainTestData()

## To show the width height and channel of images. Some of the original images have 4 channels. 
## That is why there was a need to segment them.

def calculateImageSize():
    train_df['imgHeight'] = 0
    train_df['imgWidth'] = 0
    train_df['imgChannel'] = 0

    #get train image shapes
    for i in range(len(train_df)):
        image = readImage(train_df.filepath.values[i])
        train_df.loc[i,'imgHeight'] = image.shape[0]
        train_df.loc[i,'imgWidth'] = image.shape[1]
        train_df.loc[i,'imgChannel'] = image.shape[2]

    test_df['imgHeight'] = 0
    test_df['imgWidth'] = 0
    test_df['imgChannel'] = 0

    # get test image shapes
    for i in range(len(test_df)):
        image = readImage(test_df.filepath.values[i])
        test_df.loc[i,'imgHeight'] = image.shape[0]
        test_df.loc[i,'imgWidth'] = image.shape[1]
        test_df.loc[i,'imgChannel'] = image.shape[2]
        
        
if calculate_image_size:
    print('Calculating image size')
    calculateImageSize()


        
## To show images of all species of train data
if plotImages:

    fig = plt.figure(1, figsize=(num_species, num_species))
    grid = mpl_toolkits.axes_grid1.ImageGrid(fig, 111, nrows_ncols=(num_species, num_species), 
                                             axes_pad=0.05)
    i = 0
    for species_id, sp in enumerate(species):
        for filepath in train_df[train_df['species'] == sp]['filepath'].values[:num_species]:
            ax = grid[i]
            img = readAndResizeimage(filepath, (299, 299))
            ax.imshow(img.astype(np.uint8))
            ax.axis('off')
            if i % num_species == num_species - 1:
                ax.text(250, 112, sp, verticalalignment='center')
            i += 1
    plt.show();


####################################################################[8]
    
## To show test images 

if plotImages:

    fig = plt.figure(1, figsize=(10, 10))
    grid = mpl_toolkits.axes_grid1.ImageGrid(fig, 111, nrows_ncols=(5, 10), 
                                             axes_pad=0.05)
    i = 0
    for j in range(5):
        for filepath in test_df['filepath'].values[j*5:j*5+10]:
            ax = grid[i]
            img = readAndResizeimage(filepath, (299, 299))
            ax.imshow(img.astype(np.uint8))
            ax.axis('off')
            i += 1
    plt.show();
    
    
# To show original and it's corresponding segmented image
if plotImages:
    for i in range(2):
 
        img, image_sharpen = createSegmentedImage(
            train_df.loc[i,'filepath'],(299,299))
        
        fig, axs = plt.subplots(1, 4, figsize=(20, 20))
        axs[0].imshow(img.astype(np.uint8))
        axs[1].imshow(image_sharpen.astype(np.uint8))

