import numpy as np
import time
import cv2
import pyautogui

def run(cnn_instance, minutes=5):
    while(True):
        image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR) 
        cv2.imwrite("screenshot.png", image)
        cnn_instance.test(image.flatten())
        time.sleep(minutes*60)