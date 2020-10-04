import time
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

import main

class convolutional_neural_network():
    class_names = ["Good", "Bad"]

    def __init__(self):
        self.np_im1 = np.empty([2100,3360,3])
        self.np_im2 = np.empty([2100,3360,3])

    def test(self, image):
        pass

    def data_prep(self):
        im1 = Image.open("train_image1.png")
        self.np_im1 = np.asarray(im1)/255.0
        self.np_im1=self.np_im1[:,:,:-1]

        im2 = Image.open("train_image2.png")
        self.np_im2 = np.asarray(im2)/255.0
        self.np_im2=self.np_im2[:,:,:-1]

    def run(self):
        train_images = [self.np_im1, self.np_im2]
        train_labels = [0,1]

        # print("\nTotal Time Used")
        # if time > 60:
        #     print("Minutes: %s\n\n" % round((time/60),2))
        # else:
        #     print("Seconds: %s\n\n" % round(time,2))