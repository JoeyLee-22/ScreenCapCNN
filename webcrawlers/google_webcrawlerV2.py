# pyright: reportUnboundVariable=false

import os
import requests
import pickle
from tqdm import tqdm
from bs4 import BeautifulSoup 
from data_prep import image_preparation, label_preparation

Google_Image = 'https://www.google.com/search?site=&tbm=isch&source=hp&biw=1873&bih=990&'

u_agnt = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
    'Accept-Encoding': 'none',
    'Accept-Language': 'en-US,en;q=0.8',
    'Connection': 'keep-alive',
}

def download_google_images(new_height, new_width, load_model):
    if not os.path.exists('dataset'): os.mkdir('dataset')
    if not os.path.exists('test_images'): os.mkdir('test_images')
    if not os.path.exists('train_images'): os.mkdir('train_images')
    
    if not os.path.exists("dataset/train_labels.pckl"):
        pickle.dump(0, open('dataset/num_train_labels.pckl', 'wb'))
    if not os.path.exists("dataset/test_labels.pckl"):
        pickle.dump(0, open('dataset/num_test_labels.pckl', 'wb'))

    scrape = not load_model
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
            while True:
                num_images_train = int(input('Enter number of images for training: '))
                break
            while True:
                num_images_test = int(input('Enter number of images for testing: '))
                break
            total_img = num_images_train+num_images_test
            if total_img <= 80:
                label_preparation(num_images_test, label, 'test_labels')
                pickle.dump(pickle.load(open('dataset/num_test_labels.pckl', 'rb'))+num_images_test, open('dataset/num_test_labels.pckl', 'wb'))
                label_preparation(num_images_train, label, 'train_labels')
                pickle.dump(pickle.load(open('dataset/num_train_labels.pckl', 'rb'))+num_images_train, open('dataset/num_train_labels.pckl', 'wb'))
                break
            print('\nTOTAL IMAGES CANNOT EXCEED 80; YOUR TOTAL: %d\n' % total_img)
            
        print('\nSearching Images....')
        
        search_url = Google_Image + 'q=' + search 
        
        response = requests.get(search_url, headers=u_agnt)
        html = response.text
        
        results = BeautifulSoup(html, 'html.parser').findAll('img', {'class': 'rg_i Q4LuWd'})

        count = 0
        imagelinks= []
        for res in results:
            try:
                link = res['data-src']
                imagelinks.append(link)
                count += 1
                if (count >= total_img):
                    break
            except KeyError:
                continue
        
        print(f'Found {len(imagelinks)} images')
        print('\nSTARTED DOWNLOADING...')

        for i, imagelink in enumerate(imagelinks):
            response = requests.get(imagelink)
            
            if i<num_images_train:
                imagename = 'train_images/' + search + str(i+1) + '.jpg'
                print('Downloading image {:>3}'.format(i+1) + '/%d ----->  train_images' % total_img)
            else:
                imagename = 'test_images/' + search + str(i+1) + '.jpg'
                print('Downloading image {:>3}'.format(i+1) + '/%d ----->  test_images' % total_img)
            with open(imagename, 'wb') as file:
                file.write(response.content)

        print('DOWNLOAD COMPLETE\n')

    image_preparation(new_height, new_width, load_model)