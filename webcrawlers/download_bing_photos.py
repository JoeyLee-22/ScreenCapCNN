# pyright: reportUnboundVariable=false

import os
import pickle
from data_prep import image_preparation, label_preparation
from bing_image_downloader import downloader

def download_bing_photos(new_height, new_width, load_model):
    if not os.path.exists('dataset'): os.mkdir('dataset')
    if not os.path.exists('test_images'): os.mkdir('test_images')
    if not os.path.exists('train_images'): os.mkdir('train_images')
    
    if not os.path.exists("dataset/train_labels.pckl"):
        pickle.dump(0, open('dataset/num_train_labels.pckl', 'wb'))
    if not os.path.exists("dataset/test_labels.pckl"):
        pickle.dump(0, open('dataset/num_test_labels.pckl', 'wb'))
    
    downloader.download('minecraft', limit=100,  output_dir='test_images', timeout=20)
    
    image_preparation(new_height, new_width, load_model)