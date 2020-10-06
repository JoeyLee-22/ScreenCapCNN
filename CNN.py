import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from tensorflow.keras import layers, models
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization
from keras.utils import to_categorical

import main

class convolutional_neural_network():
    class_names = ["Good", "Bad"]

    def __init__(self):
        self.np_im1 = np.empty([2100,3360,3])
        self.np_im2 = np.empty([2100,3360,3])

    def test(self, image):
        pass
  
    def resize(self, image, new_width):
        new_height = int(image.shape[0]/(image.shape[1]/new_width))
        new = np.empty([new_height,new_width,3])
        factor = image.shape[0]/new_height

        for row in range (image.shape[0]):
            for column in range (image.shape[1]):
                if row%factor == 0 and column%factor == 0:
                    new[int(row/factor)][int(column/factor)] = image[row][column]

        return new

    def data_prep(self):
        img = mpimg.imread('images/train_image1.jpg')[:,:,:-1]
        self.np_im1 = self.resize(img, 480)

        img = mpimg.imread('images/train_image2.jpg')[:,:,:-1]
        self.np_im2 = self.resize(img, 480)

        # f = plt.figure()
        # f.add_subplot(2,1, 1)
        # plt.imshow(self.np_im1.astype('uint8'))
        # f.add_subplot(2,1, 2)
        # plt.imshow(self.np_im2.astype('uint8'))
        # plt.show()

    def run(self):
        train_images = [self.np_im1, self.np_im2]
        train_labels = [[1,0],[0,1]]

        CNN = keras.Sequential([
            keras.layers.Flatten(input_shape=(300,480,3)),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dense(2, activation="softmax")
        ])

        start_time = time.time()
        CNN.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        CNN.fit(train_images, train_labels, epochs=10)
        end_time = time.time() - start_time

        print("\nTotal Time Used")
        if time > 60:
            print("Minutes: %s\n\n" % round((end_time/60),2))
        else:
            print("Seconds: %s\n\n" % round(end_time,2))