# pyright: reportUnboundVariable=false

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model as keras_load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from webcrawlers.google_webcrawlerV2 import download_google_images
from webcrawlers.download_flickr_photos import download_flickr_photos
from webcrawlers.download_bing_photos import download_bing_photos
from load_data import load_data

class convolutional_neural_network():
    def __init__(self, new_height, new_width):
        self.new_height = new_height
        self.new_width = new_width
        self.model_name = 'TwoClassClassificationModel'

    def classify(self, image):
        return np.argmax(self.model.predict(image))

    def run(self, epochs=10, load_model=False, save_model=False, train=True, evaluate=True, plot=True, data_prep=True, clear_data=False):        
        if load_model:
            (train_images, train_labels), (test_images, test_labels) = load_data()
            if not os.path.exists('%s.h5' % self.model_name):
                print("\nNO MODEL AVAILABLE\n")
                load_model=False
            else:
                print("\nLOADING MODEL...\n")
                self.model = keras_load_model('%s.h5' % self.model_name)
        
        if clear_data and not load_model:
            while True:
                user_input = input("\nCONFIRM DATA DELETION (y/n): ")
                if user_input=='y':
                    os.system("sh clear_data.sh")
                    break
                elif user_input=='n':
                    clear_data=False
                    break
        
        if data_prep or clear_data:
            print('\n')
            while(True):
                website = input("Scrape Google(0), Flickr(1), or Bing(2)? ")
                if (website == '0'):
                    download_google_images(self.new_height, self.new_width, load_model)
                    print('\n')
                    break
                elif (website == '1'):
                    download_flickr_photos(self.new_height, self.new_width, load_model)
                    print('\n')
                    break
                elif (website == '2'):
                    download_bing_photos(self.new_height, self.new_width, load_model)
                    print('\n')
                    break     

        if train and not load_model:
            (train_images, train_labels), (test_images, test_labels) = load_data()
                    
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
            hist = self.model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
            end_time = time.time() - start_time

            if end_time > 3600:
                print('Total Training Time: %dhr %.1fmin\n\n' % (int(end_time/3600),((end_time-int(end_time/3600)*3600)/60)))
            elif end_time > 60:
                print("Total Training Time: %dmin %.2fs\n\n" % ((end_time/60), (end_time-int(end_time/60)*60)))
            else:
                print("Total Training Time: %.2fs\n\n" % end_time)
                
        if save_model and not load_model:
            self.model.save('%s.h5' % self.model_name)

        if evaluate:
            print("TESTING")
            self.model.evaluate(test_images, test_labels)[1]
            print('\n')

        if plot and not load_model:            
            f = plt.figure()
            f.add_subplot(2,1, 1)
            plt.plot(hist.history['accuracy'], label='train accuracy')
            plt.plot(hist.history['val_accuracy'], label ='val accuracy')
            plt.title('Model Accuracy (top) and Model Loss (bottom)')     
            plt.ylabel('Accuracy')
            plt.legend(loc='lower left')
            
            f.add_subplot(2,1, 2)
            plt.plot(hist.history['loss'], label='train loss')
            plt.plot(hist.history['val_loss'], label ='val loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.ylim([0, 2])
            plt.legend(loc='lower left')
            
            plt.show()