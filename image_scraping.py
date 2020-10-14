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

def download_google_images(new_height, new_width):
    if not os.path.exists('dataset'): os.mkdir('dataset') 
    
    if not os.path.exists("dataset/train_labels.pckl"):
        pickle.dump(0, open('dataset/num_train_labels.pckl', 'wb'))
    if not os.path.exists("dataset/test_labels.pckl"):
        pickle.dump(0, open('dataset/num_test_labels.pckl', 'wb'))

    print('\n\n--- SCRAPING STARTED ---\n')
    first = True
    while True:
        if input('Continue Scraping? (y/n): ').lower() == 'n': break

        if first:
            file_choice = input('Enter the folder you want the images to go to (train/test): ')
            if file_choice == 'train':
                label_folder = 'train_labels'
                image_folder = 'train_images'
            elif file_choice == 'test':
                label_folder = 'test_labels'
                image_folder = 'test_images'
            else:
                continue
            first = False
        elif input('Same Folder? (y/n): ') == 'n':
            file_choice = input('Enter the folder you want the images to go to (train/test): ')
            if file_choice == 'train':
                label_folder = 'train_labels'
                image_folder = 'train_images'
            elif file_choice == 'test':
                label_folder = 'test_labels'
                image_folder = 'test_images'
            else:
                continue
        if not os.path.exists(image_folder): os.mkdir(image_folder)

        data = input('Enter your search keyword(s): ')
        label = int(input('Enter the label for this batch: '))
        num_images = int(input('Enter the number of images you want: '))

        print('\nSearching Images....')
        
        search_url = Google_Image + 'q=' + data 
        
        response = requests.get(search_url, headers=u_agnt)
        html = response.text
        
        b_soup = BeautifulSoup(html, 'html.parser') 
        results = b_soup.findAll('img', {'class': 'rg_i Q4LuWd'})

        count = 0
        imagelinks= []
        for res in results:
            try:
                link = res['data-src']
                imagelinks.append(link)
                count += 1
                if (count >= num_images):
                    break
            except KeyError:
                continue
        
        print(f'Found {len(imagelinks)} images')
        print('\nStarted downloading...')

        label_preparation(len(imagelinks), label, label_folder)

        if label_folder=='train_labels':
            pickle.dump(pickle.load(open('dataset/num_train_labels.pckl', 'rb'))+len(imagelinks), open('dataset/num_train_labels.pckl', 'wb'))
        else:
            pickle.dump(pickle.load(open('dataset/num_test_labels.pckl', 'rb'))+len(imagelinks), open('dataset/num_test_labels.pckl', 'wb'))

        pbar = tqdm(total=len(imagelinks))
        for i, imagelink in enumerate(imagelinks):
            response = requests.get(imagelink)
            
            imagename = image_folder + '/' + data + str(i+1) + '.jpg'
            with open(imagename, 'wb') as file:
                file.write(response.content)
            pbar.update(1)
        pbar.close()

        print('Download Complete\n')

    image_preparation(new_height, new_width)