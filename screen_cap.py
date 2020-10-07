import numpy as np
import time
import cv2
import pyautogui

def start(cnn_instance, minutes=5):
    while(True):
        og_image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        image = cnn_instance.resize(og_image, 480)
        print(cnn_instance.classify(np.array([image])))
        cv2.imwrite("images/screenshot.jpg", og_image)
        
        for secs in range (int(minutes*60)):
            time.sleep(0.98)
            print(str(int(minutes*60)-int(secs)) + "/" + str(int(minutes*60)), end='\r')