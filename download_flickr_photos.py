# pyright: reportUnboundVariable=false

import urllib
import os
import pickle
from flickrapi import FlickrAPI
from data_prep import image_preparation, label_preparation
from config import API_KEY, API_SECRET

def download_flickr_photos(new_height, new_width, load_model):
    if not os.path.exists('dataset'): os.mkdir('dataset')
    if not os.path.exists('test_images'): os.mkdir('test_images')
    if not os.path.exists('train_images'): os.mkdir('train_images')
    
    if not os.path.exists("dataset/train_labels.pckl"):
        pickle.dump(0, open('dataset/num_train_labels.pckl', 'wb'))
    if not os.path.exists("dataset/test_labels.pckl"):
        pickle.dump(0, open('dataset/num_test_labels.pckl', 'wb'))

    flickr = FlickrAPI(API_KEY, API_SECRET)
    scrape = not load_model
    first = True
    stop = False
    
    if scrape: print('\n\n\n\n--- SCRAPING STARTED ---\n')
    while scrape: 
        while True:
            user_input = input('Continue Scraping? (y/n): ').lower()
            if  user_input == 'n':
                stop = True
                break
            elif user_input == 'y':
                break
        if stop == True:
            break

        search = input('\nEnter search keyword(s): ')
        while True:
            label = int(input('Enter label for this batch (0: Productive; 1: Nonproductive): '))
            if label == 0:
                break
            elif label == 1:
                break
            
        while True:
            num_images_train = int(input('Enter number of images for training: '))
            if num_images_train > 500:
                print("INPUT CANNOT EXCEED 500")
            else:
                label_preparation(num_images_train, label, 'train_labels')
                pickle.dump(pickle.load(open('dataset/num_train_labels.pckl', 'rb'))+num_images_train, open('dataset/num_train_labels.pckl', 'wb'))
                break
        while True:
            num_images_test = int(input('Enter number of images for testing: '))
            if num_images_train > 500:
                print("INPUT CANNOT EXCEED 500")
            else:
                label_preparation(num_images_test, label, 'test_labels')
                pickle.dump(pickle.load(open('dataset/num_test_labels.pckl', 'rb'))+num_images_test, open('dataset/num_test_labels.pckl', 'wb'))
                break
        print('\n')
        total_img = num_images_train+num_images_test
        
        photos = flickr.walk(text=search,extras='url_o',license='0,1,2,3,4,5,6,7,8,9,10',per_page=50)
        urls = []
        count = 1
        for photo in photos:
            if count > total_img:
                print('\nMaximum number of images for download reached')
                break
            try:
                url=photo.get('url_o')
                urls.append(url) 
                if count <= num_images_train:
                    urllib.request.urlretrieve(url,'train_images/%s%d.jpg' % (search,count))
                    print('Downloading image {:>3}'.format(count) + '/%d ----->  train_images' % total_img)
                else:
                    urllib.request.urlretrieve(url,'test_images/%s%d.jpg' % (search,count))
                    print('Downloading image {:>3}'.format(count) + '/%d ----->  test_images' % total_img)
                count+=1
            except Exception as e:
                print('Download failure: ', e)
        print("Total images downloaded: %d\n\n" % (count-1))
    
    image_preparation(new_height, new_width, load_model)