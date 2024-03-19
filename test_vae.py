# file to test vae model
import os
import torch
import numpy as np
from vae_model import VAE
from PIL import Image
from music21 import midi
from img2midi import image2midi
import numpy as np
import py_midicsv as pm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from utils import load_latest_model


def generate():

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    #device = torch_directml.device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    z_dim = 64

    vae = VAE(z_dim).to(device)

    pattern = 'saved_results/vae_saved_*.pt'
    model = load_latest_model(pattern)
    if not model:
        print('No model found, closing...')
        exit(-1)


    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    os.chdir(output_dir)

    noise = torch.randn(1, z_dim, device=device)
    

    X = vae.decoder(noise)
    print(X.size())
    vutils.save_image(X.cpu().detach(), "midipics/output_vae.png", normalize=True, padding=2, dtype=torch.int64)

    midi = image2midi('midipics/output_vae.png')

    # mf = midi.MidiFile()

    # s = midi.translate.midiFileToStream(mf)
    # s.show('midi') 

    # # save the MIDI file to the output directory
    # output_path = os.path.join(output_dir, 'output.mid')
    # midi.save(output_path)

    print(f'Saved MIDI file')

if __name__ == '__main__':
    generate()