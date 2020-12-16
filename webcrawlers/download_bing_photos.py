# pyright: reportUnboundVariable=false

import urllib
import os
import pickle
from flickrapi import FlickrAPI
from data_prep import image_preparation, label_preparation
from config import API_KEY, API_SECRET

def download_bing_photos(new_height, new_width, load_model):
    if not os.path.exists('dataset'): os.mkdir('dataset')
    if not os.path.exists('test_images'): os.mkdir('test_images')
    if not os.path.exists('train_images'): os.mkdir('train_images')
    
    if not os.path.exists("dataset/train_labels.pckl"):
        pickle.dump(0, open('dataset/num_train_labels.pckl', 'wb'))
    if not os.path.exists("dataset/test_labels.pckl"):
        pickle.dump(0, open('dataset/num_test_labels.pckl', 'wb'))
    
    image_preparation(new_height, new_width, load_model)