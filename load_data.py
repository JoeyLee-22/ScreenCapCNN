import pickle
import numpy as np
from keras.utils import to_categorical
    
def load_data():
    train_images = pickle.load(open('dataset/train_images.pckl', 'rb'))
    test_images = pickle.load(open('dataset/test_images.pckl', 'rb'))

    f = open('dataset/train_labels.pckl', 'rb')
    train_labels = []
    while True:
        try:
            train_labels.append(pickle.load(f))
        except EOFError:
            break
    train_labels = (np.array(train_labels))
    train_labels.resize(pickle.load(open('dataset/num_train_labels.pckl', 'rb')),1)

    f =  open('dataset/test_labels.pckl', 'rb')
    test_labels = []
    while True:
        try:
            test_labels.append(pickle.load(f))
        except EOFError:
            break
    test_labels = (np.array(test_labels))
    test_labels.resize(pickle.load(open('dataset/num_test_labels.pckl', 'rb')),1)

    # print("\n\nload_data.py before to_categorical\nTrain Labels:")
    # print(train_labels)
    # print("Test Labels:")
    # print(test_labels)
    # print("\n\n")
    
    # train_labels = to_categorical((np.array(train_labels)))
    # test_labels = to_categorical((np.array(test_labels)))
    
    # print("\n\nload_data.py post to_categorical\nTrain Labels:")
    # print(train_labels)
    # print("Test Labels:")
    # print(test_labels)
    # print("\n\n")

    return (train_images,train_labels), (test_images,test_labels)