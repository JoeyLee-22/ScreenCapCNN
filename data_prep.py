import os
import pickle
import matplotlib.image as mpimg
import numpy as np
from resize import resize

def data_preparation(new_height, new_width):
    train_images = np.empty([len(os.listdir('train_images')),new_height,new_width,3])
    test_images = np.empty([len(os.listdir('test_images')),new_height,new_width,3])

    counter=0
    for filename in os.listdir('train_images'):
        img = mpimg.imread('train_images/%s' % filename)[:,:,:-1]
        train_images[counter] = resize(img, new_height, new_width)
        counter+=1

    counter=0
    for filename in os.listdir('test_images'):
        img = mpimg.imread('test_images/%s' % filename)[:,:,:-1]
        test_images[counter] = resize(img, new_height, new_width)
        counter+=1

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