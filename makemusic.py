import os
import torch
import numpy as np
from model import Generator
from PIL import Image
from music21 import midi
from img2midi import image2midi
import numpy as np
import py_midicsv as pm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch_directml

def generate():

    if(torch.cuda.is_available()):
        torch.cuda.empty_cache()
    #device = torch_directml.device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    z_dim = 256

    generator = Generator(z_dim).to(device)

    if os.path.exists('saved_results/generator_saved_prev.pt'):
        print('Loading generator...')
        generator.load_state_dict(torch.load('saved_results/generator_saved_prev.pt', map_location=device))
    else:
        print('No generator found, closing...')
        return -1

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)
    
    noise = torch.randn(1, z_dim, device=device)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))

    ax1.clear()
    ax1.axis("off")
    ax1.set_title("Generator Output")
    

    X = generator(noise)
    print(X.size())
    vutils.save_image(X.cpu().detach(), "midipics/output.png", normalize=True, padding=2, dtype=torch.int64)

    midi = image2midi('midipics/output.png')

    # mf = midi.MidiFile()

    # s = midi.translate.midiFileToStream(mf)
    # s.show('midi') 

    # # save the MIDI file to the output directory
    # output_path = os.path.join(output_dir, 'output.mid')
    # midi.save(output_path)

    print(f'Saved MIDI file')

if __name__ == '__main__':
    generate()