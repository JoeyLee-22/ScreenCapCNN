# pyright: reportUnboundVariable=false

import pickle
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical
from image_scraping import download_google_images

class convolutional_neural_network():
    def __init__(self, new_height, new_width):
        self.new_height = new_height
        self.new_width = new_width

    def classify(self, image):
        return np.argmax(self.model.predict(image))

    def load_data(self):
        train_images = pickle.load(open('dataset/train_images.pckl', 'rb'))
        test_images = pickle.load(open('dataset/test_images.pckl', 'rb'))

        train_labels = to_categorical(pickle.load(open('dataset/train_labels.pckl', 'rb')))
        test_labels = to_categorical(pickle.load(open('dataset/test_labels.pckl', 'rb')))

        return (train_images,train_labels), (test_images,test_labels)

    def run(self, epochs=10, train=True, evaluate=True, plot=True, data_prep=True, clear_data=False):
        if clear_data:
            if input("CONFIRM DATA DELETION (y/n): ")=='y':
                os.system("sh clear_data.sh")
        
        if data_prep:
            download_google_images(self.new_height, self.new_width)
            print('\n')

        if train:
            (train_images, train_labels), (test_images, test_labels) = self.load_data()

            print(train_images.shape)
            print(train_labels.shape)
            print(test_images.shape)
            print(test_labels.shape)

            train_images = train_images/255.0
            test_images = test_images/255.0

            self.model = Sequential()
            self.model.add(Conv2D(32, (5,5), activation='relu', input_shape=(self.new_height,self.new_width,3)))
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            self.model.add(Conv2D(32, (5,5), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2,2)))
            self.model.add(Flatten())
            self.model.add(Dense(1024, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(512, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dense(2, activation='softmax'))

            start_time = time.time()
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            hist = self.model.fit(train_images, train_labels, epochs=epochs)
            end_time = time.time() - start_time

            if end_time > 3600:
                print('Total Training Time: %dhr %.1fmin\n\n' % (int(end_time/3600),((end_time-int(end_time/3600)*3600)/60)))
            elif end_time > 60:
                print("Total Training Time: %dmin %.2fs\n\n" % ((end_time/60), (end_time-int(end_time/60)*60)))
            else:
                print("Total Training Time: %.2fs\n\n" % end_time)

        if evaluate:
            print("TESTING")
            self.model.evaluate(test_images, test_labels)[1]
            print('\n')

        if plot:
            f = plt.figure()
            f.add_subplot(2,1, 1)
            plt.plot(hist.history['accuracy'])
            plt.title('Model Accuracy (top) and Model Loss (bottom)')
            plt.ylabel('Accuracy')
            f.add_subplot(2,1, 2)
            plt.plot(hist.history['loss'])
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.show()