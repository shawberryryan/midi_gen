MIDI_Gen Project:

Welcome to the MIDI_Gen project! This project aims to explore the fascinating intersection of music and machine learning by converting MIDI files into images and then using these images to train different machine learning models. Our goal is to generate novel MIDI images—and therefore novel music—capturing the complexity and beauty of compositions in a wholly new way.

Project Overview:

The MIDI_Gen project is built around the concept of transforming MIDI files into a visual representation, which serves as the input for various machine learning models. These models are trained to understand and generate new images that can be converted back into MIDI files, resulting in new, generated music pieces. This approach opens up new avenues for creative music generation and offers insights into the structure of musical compositions from a machine learning perspective.

Models:

We have developed and tested the following models:

Variational Autoencoder (VAE): A model that learns the distribution of our data to generate new instances that mimic the original MIDI files' styles and patterns.
Generative Adversarial Network (GAN): Utilizes the adversarial process between two networks, a generator and a discriminator, to produce images that are indistinguishable from the original MIDI-converted images.
An additional model has been conceptualized but not yet tested:

Diffusion Model: Intended to gradually build up a distribution of MIDI images starting from noise, this model promises to add an innovative approach to generating music once implemented.

The inputs take the following form
![Alt text](/ashover_9 "Optional title")

Our GAN offered the best results, producing the following midi images after 300 epochs
![Alt text](/fake_image_epoch_300.png "Optional title")
