"""
File: mnist_gan.py
Creation Date: 2021-11-15

This file has a GAN model using MNIST data. The difference between this one and
the one defined in the notebook, is that this uses the class defined in gan.py,
and it also only trains on a single digit instead of trying to generate
all the digits 0-9.
"""
from gan import GAN

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.datasets import mnist

import numpy as np
from itertools import product
from plotly.subplots import make_subplots
import plotly.graph_objects as go

class MNIST_GAN(GAN):
    def __init__(self, real_data, input_shape: tuple):
        super().__init__(real_data, input_shape)

    """
    For subclassing, we need to implement how our generator and
    discriminator models would look like.
    """
    def make_generator(self) -> Model:
        model = Sequential()

        model.add(Dense(256, input_shape=(self.noise_length,)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.input_shape), activation='tanh'))
        model.add(Reshape(self.input_shape))

        model.summary()

        noise_layer = Input(shape=(self.noise_length,))
        img = model(noise_layer)

        return Model(noise_layer, img)

    def make_discriminator(self) -> Model:
        model = Sequential()

        model.add(Flatten(input_shape=self.input_shape))
        model.add(Dense(512))
        model.add(Dropout(0.3))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()

        image_layer = Input(shape=self.input_shape)
        valid_layer = model(image_layer)

        return Model(image_layer, valid_layer)

if __name__ == '__main__':
    (XTrain, YTrain), (XTest, YTest) = mnist.load_data()

    XTrain = (XTrain - 127.5) / 127.5

    # Let's start simple. Only extract all the FIVES
    fives = XTrain[np.where(YTrain == 5)]
    print(f'Number of "5" samples: {fives.shape[0]}')

    fig = make_subplots(rows=5, cols=5)

    for i, (r, c) in enumerate(product(range(1, 6), range(1, 6))):
        fig.add_trace(
            go.Heatmap(z=np.flip(fives[i], axis=0), colorscale='gray'),
            row=r, col=c
        )

    fig.update_layout(height=800, width=800, title_text="Numbers", coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    fig.write_image('original.png')

    gan = MNIST_GAN(real_data=fives, input_shape=(28, 28, 1))

    gan.save_gen_output((5, 5), 'initial_plots.png')

    gan.train(epochs=60, batch_size=256, save_frequency=10)
