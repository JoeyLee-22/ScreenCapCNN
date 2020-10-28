import numpy as np
import time
import cv2
import pyautogui
from resize import my_resize

def start(cnn_instance, minutes=5):
    class_names = ["Productive", "Nonproductive"]

    while(True):
        og_image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)
        result = cnn_instance.classify(np.array([my_resize(og_image, cnn_instance.new_height, cnn_instance.new_width)]))

        if result == 0:
            cv2.imwrite("screenshots/Productive.jpg", og_image)
        else:
            cv2.imwrite("screenshots/Nonproductive.jpg", og_image)
        print(class_names[result])
        
        for secs in range (int(minutes*60)):
            time.sleep(0.99)
            print(str(int(minutes*60)-int(secs)) + "/" + str(int(minutes*60)), end='\r')