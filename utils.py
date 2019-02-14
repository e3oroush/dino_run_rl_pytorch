from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import time
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import base64
import logging

IMAGE_WIDTH,IMAGE_HEIGHT=80,80
IMAGE_CHANNELS=4
ROI=300,500
class DinoSeleniumEnv(object):
    def __init__(self, chrome_driver_path, speed=0):
        init_script = "document.getElementsByClassName('runner-canvas')[0].id = 'runner-canvas'"
        options = Options()
        options.add_argument("--mute-audio")
        options.add_argument("disable-infobars")
        self._driver = webdriver.Chrome(executable_path=chrome_driver_path,chrome_options=options)
        self._driver.get("chrome://dino")
        self._driver.execute_script("Runner.config.ACCELERATION=%d"%speed)
        self._driver.execute_script(init_script)
    
    def press_up(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_UP)
    
    def press_down(self):
        self._driver.find_element_by_tag_name("body").send_keys(Keys.ARROW_DOWN)
    
    def is_crashed(self):
        return self._driver.execute_script("return Runner.instance_.crashed;")
    
    def pause_game(self):
        self._driver.execute_script("Runner.instance_.stop();")
    
    def resume_game(self):
        self._driver.execute_script("Runner.instance_.play();")
    
    def restart_game(self):
        self._driver.execute_script("Runner.instance_.restart();")
    
    def get_score(self):
        socre = self._driver.execute_script("return Runner.instance_.distanceMeter.digits")
        socre = ''.join(socre) # it returns as vector of str digits [1,0,0] as 100
        return int(socre)

    def end_game(self):
        self._driver.close()
    
    def grab_screen(self):
        #get image from canvas
        getbase64Script = "canvasRunner = document.getElementById('runner-canvas'); \
                            return canvasRunner.toDataURL().substring(22)"
        image_b64 = self._driver.execute_script(getbase64Script)
        screen = np.array(Image.open(BytesIO(base64.b64decode(image_b64))))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY) # gray scale
        screen = screen[:ROI[0],:ROI[1]]
        screen = cv2.resize(screen,(IMAGE_WIDTH,IMAGE_HEIGHT))
        return screen


def show_image():
    while True:
        screen = (yield)
        title = "fed_image"
        image = cv2.resize(screen, (800,400))
        cv2.imshow(title, image)
        cv2.waitKey(1)

logger_instances={}
def get_logger(logger_name, filename=None, logging_mode="DEBUG"):
    """
    Returns a handy logger with both printing to std output and file
    """
    log_format = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if logger_name not in logger_instances:
        logger_instances[logger_name] = logging.getLogger(logger_name)
        if filename:
            file_handler = logging.FileHandler(filename=filename)
            file_handler.setFormatter(log_format)
            logger_instances[logger_name].addHandler(file_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_format)
        logger_instances[logger_name].addHandler(console_handler)
        logger_instances[logger_name].setLevel(level=getattr(logging,logging_mode))
    return logger_instances[logger_name]
