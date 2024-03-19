# Use this script to train your model
# https://www.google.com/url?q=https://github.com/kok202/ApolloGAN&sa=D&source=docs&ust=1681933250015709&usg=AOvVaw0Xj9b8NSnYzpttGAV7gr15 
import os
import datetime
from PIL import Image
from music21 import midi
#from img2midi import image2midi
from midi2img import midi2image
import numpy as np
import py_midicsv as pm


def dataPreProcessor():
    path = 'input/midifiles'
    midiz = os.listdir(path)
    midis = []
    for file in midiz:
        midis.append(path+"/"+file)
        
    
    mf = midi.MidiFile() #check if midi was read correctly
    print(midis[0])
    mf.open(midis[0]) 
    mf.read()
    mf.close()
    s = midi.translate.midiFileToStream(mf)
    #s.show('midi') 

    
    new_dir = 'input/midipics'
    for file in midis:
       try:
            # call midi2image and save in new_dir as a png
            img = Image.fromarray(midi2image(file))
            img.save(new_dir+"/"+file.split("/")[-1].split(".")[0]+".png")
       except:
            print("Error: "+file)
            pass

# change the 106x106 images in halfpics to be 128x128 by adding black padding
def padImages():
    path = 'input/halfpics'
    halfpics = os.listdir(path)
    for file in halfpics:
        try:
            img = Image.open(path+"/"+file)
            img = img.convert('1')
            img = img.resize((128,128))
            img.save(path+"/"+file)
        except:
            print("Error: "+file)
            pass

def removePadding():
    path = 'input/halfpics'
    halfpics = os.listdir(path)
    for file in halfpics:
        try:
            img = Image.open(path+"/"+file)
            img = img.convert('1')
            img = img.resize((106,106))
            img.save(path+"/"+file)
        except:
            print("Error: "+file)
            pass

# in main call dataProcessor() to convert midi files to images
if __name__ == "__main__":
    #dataPreProcessor()
    padImages()