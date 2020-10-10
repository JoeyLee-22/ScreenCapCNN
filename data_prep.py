import os
import pickle
import matplotlib.image as mpimg
import numpy as np
from tqdm import tqdm
from resize import resize

def data_preparation(new_height, new_width):
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

    train_labels = np.array([[1],[0]])
    test_labels = np.array([[0],[1]])

    pickle.dump(train_labels, open('dataset/train_labels.pckl', 'wb'))
    pickle.dump(test_labels, open('dataset/test_labels.pckl', 'wb'))

    # f = plt.figure()
    # f.add_subplot(2,1, 1)
    # plt.imshow(self.train_images[0].astype('uint8'))
    # f.add_subplot(2,1, 2)
    # plt.imshow(self.train_images[1].astype('uint8'))
    # plt.show()

    # f = plt.figure()
    # f.add_subplot(2,1, 1)
    # plt.imshow(self.test_images[0].astype('uint8'))
    # f.add_subplot(2,1, 2)
    # plt.imshow(self.test_images[1].astype('uint8'))
    # plt.show()