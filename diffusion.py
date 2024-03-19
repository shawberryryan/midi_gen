# utilize denoising diffusion probabilistic model to 106x212x1 midi images

import torch
from torchvision import transforms, datasets
from denoising_diffusion_pytorch import Unet, GaussianDiffusion
from DataInserter import load_data
from PIL import Image


def train():
    model = Unet(
        dim = 64,
        dim_mults = (1, 2, 4, 8),
        channels=1
    )

    diffusion = GaussianDiffusion(
        model,
        image_size = 128,
        timesteps = 1000,   # number of steps
    )

    # pulling training images from input/halfpics into tensor
    path = 'input/halfpics'
    pixset, imgset = load_data(path)
    tensor = torch.from_numpy(pixset)
    print(tensor.shape)
    
    loss = diffusion(tensor)
    loss.backward()

    sampled_images = diffusion.sample(batch_size=4)

    # save sampled images
    for i, img in enumerate(sampled_images):
        img = transforms.ToPILImage()(img.clamp(0., 1.).squeeze())
        img.save(f'sample_{i}.png')


if __name__ == '__main__':
    train()