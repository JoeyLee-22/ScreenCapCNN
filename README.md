# Screen Capture Convolutional Neural Network
## A CNN with a web scraper for data, optimized resize function, and data preparation 
- - -
This project is a machine learning model that categorizes images into two categories. After it has finished training, it is used to take screenshots and classify them as either productive or nonproductive.
- - -
### Running the Program

This project was built and tested using Python 3.8. To run this program, you MUST use a version of Python 3.

TO RUN: use "python3 main.py"

TO CONFIGURE: open main.py and set variables in cnn.run() to your preferences

  - epochs(any int): number of times the CNN goes through all the training data 
  - load_model(T/F): whether or not the program loads a previously trained model. If this parameter is true, the parameters save_model, train, plot, data_prep, and clear_data will automatically be false even if they are manually set to true
  - save_model(T/F): whether or not the model is saved after training
  - train(T/F): whether or not the program trains
  - evaluate(T/F): whether or not the program tests the model on test data
  - plot(T/F): whether or not the program plots the loss and accuracy over epochs
  - data_prep(T/F): whether or not the program starts web scraping for training and testing data
  - clear_data(T/F): whether or not the program resets all the data and asks for new ones

- - - 
### Modules Used

All modules can be installed via pip

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