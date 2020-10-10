# pyright: reportUnboundVariable=false

import os
import pickle
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
from resize import resize

def image_preparation(new_height, new_width):
    train_images = np.empty([len(os.listdir('train_images')),new_height,new_width,3])
    test_images = np.empty([len(os.listdir('test_images')),new_height,new_width,3])

    print('\nPREPPING TRAIN IMAGES')
    pbar = tqdm(total=len(os.listdir('train_images')))
    counter=0
    for filename in os.listdir('train_images'):
        img = mpimg.imread('train_images/%s' % filename)
        # if img.shape[2] == 3:
        #     train_images[counter] = resize(img, new_height, new_width)
        # else:
        #     train_images[counter] = resize(img[:,:,:-1], new_height, new_width)
        train_images[counter] = resize(img, new_height, new_width)
        pbar.update(1)
        counter+=1
    pbar.close()

    print('PREPPING TEST IMAGES')
    pbar = tqdm(total=len(os.listdir('test_images')))
    counter=0
    for filename in os.listdir('test_images'):
        img = mpimg.imread('test_images/%s' % filename)[:,:,:-1]
        # if img.shape[2] == 3:
        #     test_images[counter] = resize(img, new_height, new_width)
        # else:
        #     test_images[counter] = resize(img[:,:,:-1], new_height, new_width)
        test_images[counter] = resize(img, new_height, new_width)
        pbar.update(1)
        counter+=1
    pbar.close()

    pickle.dump(train_images, open('dataset/train_images.pckl', 'wb'))
    pickle.dump(test_images, open('dataset/test_images.pckl', 'wb'))

def label_preparation(num_labels, label, type):
    train_labels = np.array([])
    test_labels = np.array([])

    if type=='train_labels':
        train_labels = np.concatenate((train_labels, np.array([label])))
        for i in range (num_labels-1):
            train_labels = np.vstack((train_labels, np.array([label])))
        pickle.dump(train_labels, open('dataset/train_labels.pckl', 'ab'))
    elif type=='test_labels':
        test_labels = np.concatenate((test_labels, np.array([label])))
        for i in range (num_labels-1):
            test_labels = np.vstack((test_labels, np.array([label])))
        pickle.dump(test_labels, open('dataset/test_labels.pckl', 'ab'))  