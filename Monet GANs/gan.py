"""
File: gan.py
Creation Date: 2021-11-08

This file contains the class implementation of a
General Adversarial Network (GAN). The reason a class
implementation is necessary, is because a GAN is two
separate networks that require special training steps
instead of the usual .fit(). One model is trained on a batch,
then held frozen while the other is trained for a batch,
and this process repeats.

Source for general structure of class:
https://wiki.pathmind.com/generative-adversarial-network-gan
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import numpy as np
import matplotlib.pyplot as plt

class GAN:
    def __init__(self, real_data, input_shape: tuple, noise_length=100):
        # The data can be whatever format...
        # There is no preprocessing that this class will do.
        # It should be ready to just feed in the model.
        self.real_data = real_data
        self.input_shape = input_shape
        self.noise_length = noise_length
        # Make the optimizer. These parameters are good for GANs
        self.optimizer = Adam(2e-4, 0.5)
        # Build the discriminator and compile it with binary cross entropy
        self.discriminator = self.make_discriminator()
        self.discriminator.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        # Same with our generator
        self.generator = self.make_generator()
        self.generator.compile(optimizer=self.optimizer, loss='binary_crossentropy')

        # Now, we need to chain the generator and discriminator together
        # to create a new model. We also need to freeze the discriminator
        # weights to prevent them from being influenced by the generator.
        # The input to the combined model is the input to the generator
        inp = Input(shape=(self.noise_length,))
        gen_output = self.generator(inp)
        # Freeze the weights in the discriminator
        self.discriminator.trainable = False
        disc_output = self.discriminator(gen_output)
        # Build the combined model
        self.combined_model = Model(inp, disc_output)
        self.combined_model.compile(optimizer=self.optimizer, loss='binary_crossentropy')

    def make_generator(self) -> Model:
        raise NotImplementedError('Please implement your generator model.')

    def make_discriminator(self) -> Model:
        raise NotImplementedError('Please implement your discriminator model.')

    def train(self, epochs, batch_size, save_frequency=5):
        """
        The main train method. This will alternate training
        the discriminator and generator. For the discriminator,
        it will take a batch of real images, and fake images
        generated using the generator.
        For training the generator, the label for these images will
        be real, since we are tyring to fool the discriminator.
        :param epochs: The number of epochs to train for. One epoch
        constitutes one pass through the real data to train the discriminator,
        and training the generator for the same number of images.
        :param save_frequency: How often to save the combined model
        :return:
        """
        amount_batches = self.real_data.shape[0] // batch_size + 1
        for epoch in range(1, epochs + 1):
            print(f'Epoch: {epoch}')
            for batch_num in range(amount_batches):
                # Index the array to get the current batch.
                real_batch_data = self.real_data[batch_size * batch_num: batch_size * (batch_num + 1)]
                # Pass random noise through the generator data, and label these
                fake_gen_data = self.generator.predict(np.random.normal(0, 1, size=(batch_size, self.noise_length)))
                # Join the two together
                # Also build the real and fake labels
                discriminator_input = np.vstack((real_batch_data, fake_gen_data))
                discriminator_labels = np.concatenate((np.ones(batch_size), np.zeros(batch_size)))[:, np.newaxis]
                # Train on batch with the combined data
                disc_loss = self.discriminator.train_on_batch(discriminator_input, discriminator_labels)

                # Now to train the generator, use the combined model so the
                # discriminator weights are frozen.
                # Make some noise, BUT LABEL THESE AS TRUE!
                gen_input = np.random.normal(0, 1, size=(batch_size, self.noise_length))
                gen_output = np.ones((batch_size, 1))

                gen_loss = self.generator.train_on_batch(gen_input, gen_output)

                print(f'D Loss: {disc_loss[0]}, D Acc: {disc_loss[1] * 100}, '
                      f'G Loss: {gen_loss[0]}, G Acc: {gen_loss[1] * 100}')

    def save_gen_output(self, samples, plot_shape: tuple, filename=''):
        """
        Pass random noise samples through the generator
        and outputs to a file.
        :param samples:
        :param plot_shape:
        :param filename:
        :return:
        """
        if plot_shape[0] * plot_shape[1] != samples:
            raise ValueError("Plot shape doesn't match number of samples.")
        r, c = plot_shape
        noise = np.random.normal(0, 1, (r * c, self.noise_length))
        gen_input = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_input = 0.5 * gen_input + 0.5

        fig, axes = plt.subplots(r, c)
        count = 0
        for i in range(r):
            for j in range(c):
                axes[i, j].imshow(gen_input[count, :, :, 0], cmap='gray')
                axes[i, j].axis('off')
                count += 1
        if filename == '':
            filename = 'gen_output.png'
        if filename[-4:] != '.png':
            filename += '.png'
        fig.savefig(filename)
        plt.close()





