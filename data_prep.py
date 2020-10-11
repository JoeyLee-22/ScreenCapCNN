# pyright: reportUnboundVariable=false

import os
import pickle
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
from PIL import Image
from resize import my_resize

def image_preparation(new_height, new_width):
    train_images = np.empty([len(os.listdir('train_images')),new_height,new_width,3])
    test_images = np.empty([len(os.listdir('test_images')),new_height,new_width,3])

    print('\nPREPPING TRAIN IMAGES')
    pbar = tqdm(total=len(os.listdir('train_images')))
    counter=0
    for filename in os.listdir('train_images'):
        img = mpimg.imread('train_images/%s' % filename)
        if img.shape[0] < new_height or img.shape[1] < new_width:
            train_images[counter] = np.array(Image.open('train_images/%s' % filename).resize((new_width, new_height)).convert("RGB"))
        else:
            train_images[counter] = my_resize(img, new_height, new_width)
        pbar.update(1)
        counter+=1
    pbar.close()

    print('PREPPING TEST IMAGES')
    pbar = tqdm(total=len(os.listdir('test_images')))
    counter=0
    for filename in os.listdir('test_images'):
        img = mpimg.imread('test_images/%s' % filename)
        if img.shape[0] < new_height or img.shape[1] < new_width:
            test_images[counter] = np.array(Image.open('test_images/%s' % filename).resize((new_width, new_height)).convert("RGB"))
        else:
            test_images[counter] = my_resize(img, new_height, new_width)
        pbar.update(1)
        counter+=1
    pbar.close()

    pickle.dump(train_images, open('dataset/train_images.pckl', 'wb'))
    pickle.dump(test_images, open('dataset/test_images.pckl', 'wb'))

def label_preparation(num_labels, label, type):
    train_labels = np.array([])
    test_labels = np.array([])

    if not os.path.exists("dataset/train_labels.pckl"):
        pickle.dump(0, open('dataset/num_train_labels.pckl', 'wb'))
    if not os.path.exists("dataset/test_labels.pckl"):
        pickle.dump(0, open('dataset/num_test_labels.pckl', 'wb'))

    if type=='train_labels':
        pickle.dump(pickle.load(open('dataset/num_train_labels.pckl', 'rb'))+num_labels, open('dataset/num_train_labels.pckl', 'wb'))
        train_labels = np.concatenate((train_labels, np.array([label])))
        for i in range (num_labels-1):
            train_labels = np.vstack((train_labels, np.array([label])))
        pickle.dump(train_labels, open('dataset/train_labels.pckl', 'ab'))
    else:
        pickle.dump(pickle.load(open('dataset/num_test_labels.pckl', 'rb'))+num_labels, open('dataset/num_test_labels.pckl', 'wb'))
        test_labels = np.concatenate((test_labels, np.array([label])))
        for i in range (num_labels-1):
            test_labels = np.vstack((test_labels, np.array([label])))
        pickle.dump(test_labels, open('dataset/test_labels.pckl', 'ab'))  