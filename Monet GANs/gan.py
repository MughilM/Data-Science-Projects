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
import sys

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from itertools import product

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
                real_batch_data = np.expand_dims(self.real_data[batch_size * batch_num: batch_size * (batch_num + 1)],
                                                 axis=-1)
                # Pass random noise through the generator data, and label these
                fake_gen_data = self.generator.predict(np.random.normal(0, 1, size=(len(real_batch_data), self.noise_length)))
                # Join the two together
                # Also build the real and fake labels
                discriminator_input = np.vstack((real_batch_data, fake_gen_data))
                discriminator_labels = np.concatenate((np.ones(real_batch_data.shape[0]), np.zeros(fake_gen_data.shape[0])))[:, np.newaxis]
                # Train on batch with the combined data
                disc_loss = self.discriminator.train_on_batch(discriminator_input, discriminator_labels)

                # Now to train the generator, use the combined model so the
                # discriminator weights are frozen.
                # Make some noise, BUT LABEL THESE AS TRUE!
                gen_input = np.random.normal(0, 1, size=(batch_size * 2, self.noise_length))
                gen_output = np.ones((batch_size * 2, 1))

                gen_loss = self.combined_model.train_on_batch(gen_input, gen_output)

                print(disc_loss, gen_loss)
            if epoch % save_frequency == 0:
                self.save_gen_output((5, 5), filename=f'epoch_{epoch}.png')

    def save_gen_output(self, plot_shape: tuple, filename=''):
        """
        Pass random noise samples through the generator
        and outputs to a file.
        :param plot_shape: A 2-tuple showing number of images.
        :param filename:
        :return:
        """
        rows, columns = plot_shape
        # Pass proper number of samples through generator
        noise = np.random.normal(0, 1, (rows * columns, self.noise_length))
        # We don't want the final dimension of size 1 (reflects a single channel)
        gen_output = np.squeeze(self.generator.predict(noise))
        # Convert -1 to 1 range to 0 to 1 range
        gen_output = 0.5 * gen_output + 0.5

        fig = make_subplots(rows=rows, cols=columns)

        for i, (r, c) in enumerate(product(range(1, rows + 1), range(1, columns + 1))):
            fig.add_trace(
                go.Heatmap(z=np.flip(gen_output[i], axis=0), colorscale='gray'),
                row=r, col=c
            )

        fig.update_layout(height=800, width=800, title_text="Numbers", coloraxis_showscale=False)
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)

        fig.write_image(filename)
