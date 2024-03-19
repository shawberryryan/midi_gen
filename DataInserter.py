import os
from PIL import Image
from matplotlib import pyplot as plt 
import numpy as np

def access_images(img_list,path):
    pixels = []
    imgs = []
    for i in range(len(img_list)):
        if 'png' in img_list[i]:
            try:
                img = Image.open(path+'/'+img_list[i],'r')
                img = img.convert('1')
                pix = np.array(img.getdata())
                pix = pix.astype('float32')
                pix /= 255.0
                pixels.append(pix.reshape(1,106,212))
                imgs.append(img)
            except Exception as e:
                print("Error: "+str(e))
                pass
    return np.array(pixels),imgs

def show_image(pix_list):
    array = np.array(pix_list.reshape(106,106), dtype=np.uint8)
    new_image = Image.fromarray(array)
    new_image.show()

def load_data(path):
    os.getcwd()
    img_list = os.listdir(path)
        
    return access_images(img_list,path)
