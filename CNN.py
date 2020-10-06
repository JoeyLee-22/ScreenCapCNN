import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import image
from tensorflow.keras import layers, models

import main

class convolutional_neural_network():
    class_names = ["Good", "Bad"]

    def __init__(self):
        self.np_im1 = np.empty([2100,3360,3])
        self.np_im2 = np.empty([2100,3360,3])

    def test(self, image):
        pass

    def data_prep(self):
        self.np_im1 = mpimg.imread('images/train_image1.png')[:,:,:-1]
        self.np_im2 = mpimg.imread('images/train_image2.png')[:,:,:-1]

        print(self.np_im1.shape)

        # plt.imshow(self.np_im2)
        # plt.show()

    def run(self):
        train_images = [self.np_im1, self.np_im2]
        train_labels = [[1,0],[0,1]]

        model = models.Sequential()
        model.add(layers.Conv2D(64, kernel_size=3, activation='relu', input_shape=(None,None,3)))
        model.add(layers.Conv2D(32, kernel_size=3, activation='relu'))
        model.add(layers.Flatten())
        model.add(layers.Dense(10, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_images, train_labels, epochs=10)

        # print("\nTotal Time Used")
        # if time > 60:
        #     print("Minutes: %s\n\n" % round((time/60),2))
        # else:
        #     print("Seconds: %s\n\n" % round(time,2))