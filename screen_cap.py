import numpy as np
import time
import cv2
import pyautogui
from resize import my_resize

def start(cnn_instance, new_height, new_width, minutes=5):
    class_names = ["Good", "Bad"]

    while(True):
        og_image = cv2.cvtColor(np.array(pyautogui.screenshot()), cv2.COLOR_RGB2BGR)

        result = cnn_instance.classify(np.array([my_resize(og_image, new_height, new_width)]))

        # if result == 0:
        #     cv2.imwrite("images/good.jpg", og_image)
        # else:
        #     cv2.imwrite("images/bad.jpg", og_image)
        print(class_names[result])
        
        for secs in range (int(minutes*60)):
            time.sleep(0.98)
            print(str(int(minutes*60)-int(secs)) + "/" + str(int(minutes*60)), end='\r')