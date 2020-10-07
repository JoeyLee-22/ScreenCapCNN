import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.utils import to_categorical

class convolutional_neural_network():
    class_names = ["Good", "Bad"]

    def classify(self, image):
        predictions = self.model.predict(image)
        max = np.argmax(predictions)
        return self.class_names[max]

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
        np_im1 = self.resize(img, 480)

        img = mpimg.imread('images/train_image2.jpg')[:,:,:-1]
        np_im2 = self.resize(img, 480)

        img = mpimg.imread('images/test_image1.jpg')[:,:,:-1]
        np_im3 = self.resize(img, 480)

        img = mpimg.imread('images/test_image2.jpg')[:,:,:-1]
        np_im4 = self.resize(img, 480)

        self.train_images = np.array([np_im1, np_im2])/255.0
        self.train_labels = np.array([[0],[1]])
        self.test_images = np.array([np_im3, np_im4])/255.0
        self.test_labels = np.array([[0],[1]])

        self.train_labels_one_hot = to_categorical(self.train_labels)
        self.test_labels_one_hot = to_categorical(self.test_labels)

        # f = plt.figure()
        # f.add_subplot(2,1, 1)
        # plt.imshow(self.np_im1.astype('uint8'))
        # f.add_subplot(2,1, 2)
        # plt.imshow(self.np_im2.astype('uint8'))
        # plt.show()

    def run(self, epochs):
        self.model = Sequential()

        self.model.add(Conv2D(32, (5,5), activation='relu', input_shape=(300,480,3)))
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
        hist = self.model.fit(self.train_images, self.train_labels_one_hot, epochs=epochs)
        end_time = time.time() - start_time

        if end_time > 60:
            print("Total Training Time: %smins\n\n" % round((end_time/60),2))
        else:
            print("Total Training Time: %ss\n\n" % round(end_time,2))

        self.model.evaluate(self.test_images, self.test_labels_one_hot)[1]
        print('\n')

        # plt.plot(hist.history['accuracy'])
        # plt.title('Model Accuracy')
        # plt.xlabel('Epoch')
        # plt.ylabel('Accuracy')
        # plt.show()

        # plt.plot(hist.history['loss'])
        # plt.title('Model Loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.show()