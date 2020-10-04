import numpy as np
import time
import cv2
import pyautogui
from CNN import convolutional_neural_network

def run(minutes=5):
    cnn = convolutional_neural_network()
    while(True):
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR) 
        # cv2.imwrite("screenshot.png", image)
        cnn.test(image.flatten())
        time.sleep(minutes*60)