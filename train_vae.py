# file to train the VAE model from vae_model.py
# data is taken from midipics folder

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch_directml
from DataInserter import load_data
from vae_model import VAE
from img2midi import image2midi
from utils import load_latest_model

def weighted_bce_loss(output, target, pos_weight=10.0):
        #custom loss function that penalizes missed notes (white pixels) more than black space

        if not (target.size() == output.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), output.size()))
        weight = target * pos_weight + (1 - target)

        loss = F.binary_cross_entropy(output, target, weight=weight, reduction='none')

        return loss.mean()

def train_loop():
    # Define the device to be used for training
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch_directml.device()

    # Define the hyperparameters
    batch_size = 128
    z_dim = 256
    num_epochs = 200
    lr = 1e-3
    beta1 = 0.5
    bce_weight = 5

    # Create the VAE
    vae = VAE(z_dim).to(device)

    # Load the latest model
    pattern = 'saved_results/vae_saved_*.pt'
    model, latest_file = load_latest_model(pattern)
    init_epoch = 0
    if model:
        vae.load_state_dict(model)
        init_epoch = int(latest_file.split('_')[-1].split('.')[0])
        print('Initializing training from epoch '+str(init_epoch) )
    else:
        print('No model found, creating new model...')


    # Define the dataset
    path = 'input/midipics'
    pixset, imgset = load_data(path)
    # Define the dataloader
    dataloader = DataLoader(pixset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Define the loss function and optimizer 
    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=0.01)


    # Create a list to store the losses for each epoch
    losses = []

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    plt.show(block=False)

    save_dir = "saved_results"
    fixed_noise = torch.randn(64, z_dim, device=device)
    # Start training
    end_training = False
    for epoch in range(num_epochs):
        print('Training Epoch: '+str(epoch))
        for i, data in enumerate(dataloader):
            # Get batch of data and reshape into a vector
            data = data.to(device)

            # Forward pass through the VAE
            x_hat, mu, log_var = vae.forward(data)

            # Calculate the reconstruction loss and kl divergence
            recon_loss = weighted_bce_loss(x_hat, data, pos_weight=bce_weight)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

            # Calculate the total loss and perform backprop
            loss = recon_loss + kl_div
            vae_optimizer.zero_grad()
            loss.backward()
            vae_optimizer.step()

            # Store the loss
            losses.append(loss.item())

            # Print the loss every 100 batches
            if i % 100 == 0:
                print('Epoch: '+str(epoch)+' Batch: '+str(i)+' Loss: '+str(loss.item()))

        # Save the reconstructed images from the fixed noise
        with torch.no_grad():
            fake = vae.decoder(fixed_noise).detach().cpu()
        ax1.clear()
        ax1.axis("off")
        ax1.set_title("Decoder Output at Epoch "+str(epoch))
        ax1.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))

        # Plot the generator and discriminator losses on the second subplot
        ax2.clear()
        ax2.set_title("Loss During Training")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.plot(losses,label="loss")
        ax2.legend()

        # Show the plot every 10 epochs
        if epoch % 10 == 0:
            plt.draw()
            plt.pause(1)
        # Save the model every 10 epochs
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_saved_{epoch+init_epoch+1}.pt"))
        # Save an output sample of the decoder every 10 epochs with the name of the epoch
            with torch.no_grad():
                noise = torch.randn(1, z_dim, device=device)
                fake = vae.decoder(noise).detach().cpu()
                vutils.save_image(fake.cpu().detach(), os.path.join(save_dir, f"train_{epoch+init_epoch+1}.png"), normalize=True, padding=2, dtype=torch.int64)
                try:
                    midi = image2midi(os.path.join(save_dir, f"train_{epoch+init_epoch+1}.png"))
                except:
                    print('Error converting image to midi')
                    pass
    # save the model
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_saved_{epoch+init_epoch+1}.pt"))

    # Save an output sample of the decoder
    with torch.no_grad():
        noise = torch.randn(1, z_dim, device=device)
        fake = vae.decoder(noise).detach().cpu()
        vutils.save_image(fake.cpu().detach(), "train.png", normalize=True, padding=2, dtype=torch.int64)
        midi = image2midi('train.png')


if __name__ == "__main__":
    train_loop()