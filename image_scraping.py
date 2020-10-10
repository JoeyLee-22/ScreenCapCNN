# pyright: reportUnboundVariable=false

import os
import requests 
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

print('\n\n--- SCRAPING STARTED ---\n')

def download_google_images(new_height, new_width):
    first = True
    while True:
        if input('Continue? (y/n): ').lower() == 'n': break

        if first:
            if input('Enter the folder you want the images to go to (train/test): ') == 'train':
                folder = 'train_images'
            else:
                folder = 'test_images'
            first = False
        elif input('Same Folder? (y/n): ') == 'n':
            if input('Enter the folder you want the images to go to (train/test): ') == 'train':
                folder = 'train_images'
            else:
                folder = 'test_images'
        if not os.path.exists(folder): os.mkdir(folder)

        data = input('Enter your search keyword: ')
        label = int(input('Enter the label for this batch: '))
        num_images = int(input('Enter the number of images you want: '))

        label_preparation(num_images, label, folder)
        
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

        pbar = tqdm(total=len(imagelinks))
        for i, imagelink in enumerate(imagelinks):
            response = requests.get(imagelink)
            
            imagename = folder + '/' + data + str(i+1) + '.jpg'
            with open(imagename, 'wb') as file:
                file.write(response.content)
            pbar.update(1)
        pbar.close()

        print('Download Complete\n')

    image_preparation(new_height, new_width)