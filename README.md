# Screen Capture Convolutional Neural Network
## A CNN with a flexible web scraper, optimized resize function, and data preparation 
- - -
This project is a machine learning model that categorizes images into categories. It is used to take screenshots and classify them as either productive or nonproductive.
- - -
### Running the Program

This project was built and tested using Python 3.8. To run this program, you MUST use a version of Python 3.

TO RUN: use "python3 main.py"

TO CONFIGURE: open main.py and set variables in cnn.run() to your preferences

  - epochs: number of times the CNN goes through all the training data
  - train: whether or not the program trains
  - evaluate: whether or not the program tests the model on test data
  - plot: whether or not the program plots the loss and accuracy over epochs
  - data_prep: whether or not the program starts web scraping for training and testing data
  - clear_data: whether or not the program resets all the data and asks for new ones

- - - 
### Libraries Used
  - numpy
  - time
  - cv2
  - pyautogui
  - os
  - requests
  - tqdm
  - bs4
  - pickle
  - matplotlib
  - PIL