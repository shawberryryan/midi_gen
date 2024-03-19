import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch_directml
from model import Generator, Discriminator
from DataInserter import load_data
from PIL import Image

def train_loop():
    # Define the device to be used for training
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch_directml.device()

    # Define the hyperparameters
    batch_size = 128
    z_dim = 256
    num_epochs = 300
    lr = 1e-3
    beta1 = 0.5

    # Create the discriminator and generator, if one exists in saved_results, load it
    discriminator = Discriminator().to(device)
    generator = Generator(z_dim).to(device)

    if os.path.exists('saved_results/discriminator_saved.pt'):
        print('Loading discriminator...')
        discriminator.load_state_dict(torch.load('saved_results/discriminator_saved.pt', map_location=device))
    else:
        print('No discriminator found, creating new discriminator...')


    if os.path.exists('saved_results/generator_saved.pt'): #change this
        print('Loading generator...')
        generator.load_state_dict(torch.load('saved_results/generator_saved.pt', map_location=device))
    else:
        print('No generator found, creating new generator...')

    # Define the dataset
    path = 'input/midipics'
    pixset, imgset = load_data(path)
    # Define the dataloader
    dataloader = DataLoader(pixset, batch_size=batch_size, shuffle=True, num_workers=8)

    # Define the loss function and optimizer for the discriminator and generator
    criterion = nn.BCELoss()
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr/4, betas=(beta1, 0.999), weight_decay=0.01)
    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999), weight_decay=0.01)

    # Create a fixed noise vector to visualize the generator output during training
    fixed_noise = torch.randn(64, z_dim, device=device)

    # Create a list to store the generator and discriminator losses for each epoch
    g_losses = []
    d_losses = []

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
    plt.show(block=False)

    save_dir = "saved_results"
    
    # Start training
    for epoch in range(num_epochs):
        print('Training Epoch: '+str(epoch))
        for i, data in enumerate(dataloader):
            # Update the discriminator
            discriminator.zero_grad()
            real_data = data.to(device)
            batch_size = real_data.size(0)
            label_real = torch.full((batch_size,), 1, device=device, dtype=torch.float32)
            label_fake = torch.full((batch_size,), 0, device=device, dtype=torch.float32)
            
            # Train the discriminator on real data
            output_real = discriminator(real_data)
            d_loss_real = criterion(output_real, label_real)
            d_loss_real.backward()
            
            # Train the discriminator on fake data
            noise = torch.randn(batch_size, z_dim, device=device)
            fake_data = generator(noise)
            output_fake = discriminator(fake_data.detach())
            d_loss_fake = criterion(output_fake, label_fake)
            d_loss_fake.backward()
            
            # Update the discriminator's parameters
            d_loss = d_loss_real + d_loss_fake
            d_optimizer.step()
            
            # Update the generator
            generator.zero_grad()
            label_real = torch.full((batch_size,), 1, device=device, dtype=torch.float32)
            output_fake = discriminator(fake_data)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            
            # Update the generator's parameters
            g_optimizer.step()
            
            # Record the losses
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())    
        print('Epoch #'+str(epoch)+' Losses (G,D): '+str(g_losses[-1])+', '+str(d_losses[-1]))
        # Plot the generator output on the first subplot
        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        ax1.clear()
        ax1.axis("off")
        ax1.set_title("Generator Output at Epoch "+str(epoch))
        ax1.imshow(np.transpose(vutils.make_grid(fake, padding=2, normalize=True),(1,2,0)))

        # Plot the generator and discriminator losses on the second subplot
        ax2.clear()
        ax2.set_title("Generator and Discriminator Loss During Training")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.plot(g_losses,label="Generator")
        ax2.plot(d_losses,label="Discriminator")
        ax2.legend()

        # Show the plot every 10 epochs
        if epoch % 10 == 0:
            plt.draw()
            plt.pause(1)

        # save the model and a plot every 100 epochs
        if epoch % 10 == 0 and epoch != 0:
            torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_saved_{epoch}.pt"))
            torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_saved_{epoch}.pt"))
            vutils.save_image(fake, os.path.join(save_dir, f"fake_image_epoch_{epoch}.png"))

    # Create a folder to save the images and models
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the loss plot
    fig = plt.figure(figsize=(8, 8))
    plt.plot(g_losses,label="Generator")
    plt.plot(d_losses,label="Discriminator")
    plt.title("Generator and Discriminator Loss During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"loss_plot_{epoch}.png"))
    
    # Save the fake image
    fake_image = vutils.make_grid(fake, padding=2, normalize=True)
    vutils.save_image(fake_image, os.path.join(save_dir, f"fake_image_epoch_{epoch}.png"))

    # Save the discriminator and generator models
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f"discriminator_saved.pt"))
    torch.save(generator.state_dict(), os.path.join(save_dir, f"generator_saved.pt"))


if __name__ == "__main__":
    train_loop()