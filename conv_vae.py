#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-01 15:18:38
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$


import tensorflow as tf
from ops import *
import numpy as np
import matplotlib.pyplot as plt


class ConvVAE(object):
    """ Convolutional variational autoencoder """

    def __init__(self, latent_dim, batch_size=64):
        self.latent_dim = latent_dim
        self.batch_size = batch_size

        # placeholder for input images. Input images are RGB 64x64
        self.input_images = tf.placeholder(tf.float32, [None, 144, 256, 3])
        input_images_flat = tf.reshape(self.input_images, [-1, 144 * 256 * 3])

        # placeholder for z_samples. We are using this placeholder when we are generating new images
        self.z_samples = tf.placeholder(tf.float32, [None, self.latent_dim])

        # encoder
        z_mean, z_logstd = self.encoder()

        # decoder input
        samples = tf.random_normal([self.batch_size, self.latent_dim], 0, 1, dtype=tf.float32)
        z = z_mean + (tf.exp(.5 * z_logstd) * samples)

        # decoder
        self.generated_images = self.decoder(z)
        self.generated_images_sigmoid = tf.sigmoid(self.generated_images)
        generated_images_flat = tf.reshape(self.generated_images, [-1, 144 * 256 * 3])

        # let's calculate the loss self.generation_loss = -tf.reduce_sum(input_images_flat * tf.log(1e-8 +
        # generated_images_flat)+ (1 - input_images_flat) * tf.log(1e-8 + 1 - generated_images_flat), 1)

        self.generation_loss = tf.reduce_sum(
            tf.maximum(generated_images_flat, 0) - generated_images_flat * input_images_flat \
            + tf.log(1 + tf.exp(-tf.abs(generated_images_flat))), 1)

        self.latent_loss = 0.5 * tf.reduce_sum(tf.square(z_mean) + tf.exp(2 * z_logstd) - 2 * z_logstd - 1, 1)

        self.loss = tf.reduce_mean(self.generation_loss + self.latent_loss)

        # and our optimizer
        learning_rate = 1e-3
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        # generator for new frames
        self.generator = self.decoder(self.z_samples, activation=tf.nn.sigmoid, reuse=True)

    def encoder(self):
        # first convolutional layer 144x256x3 -> 72x128x16
        h1 = tf.nn.relu(conv2d(self.input_images, 3, 16, 'conv1'))

        # second convolutional layer 72x128x16 -> 36x64x32
        h2 = tf.nn.relu(conv2d(h1, 16, 32, 'conv2'))

        # fully connected layer 36x64x32 -> (-1, 36x64x32) -> (-1, 18x32x32)
        h2_flat = tf.reshape(h2, [-1, 36 * 64 * 32])
        h3_flat = dense(h2_flat, 36 * 64 * 32, 18 * 32 * 32, 'h3_flat_dense')
        z_mean = dense(h3_flat, 18 * 32 * 32, self.latent_dim, 'z_mean_dense')
        z_logstd = dense(h3_flat, 18 * 32 * 32, self.latent_dim, 'z_stddev_dense')

        return z_mean, z_logstd

    def decoder(self, z, activation=tf.identity, reuse=False):
        # fully connected layer latent_dim -> 18x32x32
        z_fc1 = dense(z, self.latent_dim, 18 * 32 * 32, 'z_fc_dense1', reuse)
        # fully connected layer 18x32x32 => 36*64*32
        z_fc2 = dense(z_fc1, 18 * 32 * 32, 36 * 64 * 32, 'z_fc_dense2', reuse)

        # first deconvolutional layer 36x64x32 -> 72x128x32
        z_matrix = tf.nn.relu(tf.reshape(z_fc2, [-1, 36, 64, 32]))
        h1 = tf.nn.relu(deconv2d(z_matrix, [self.batch_size, 72, 128, 32], 'deconv1', reuse=reuse))

        # second deconvolutional layer 32x64x32 -> 72x128x16
        h2 = tf.nn.relu(deconv2d(h1, [self.batch_size, 144, 256, 3], 'deconv2', reuse=reuse))

        # third deconvolutional layer 32x64x32 -> 144x256x3 上采样
        h3 = deconv2d(h2, [self.batch_size, 288, 512, 3], 'deconv3', reuse=reuse)

        return activation(h2)

    def training_step(self, sess, input_images):
        sess.run(self.optimizer, feed_dict={self.input_images: input_images})

    def loss_step(self, sess, input_images):
        return sess.run(self.loss, feed_dict={self.input_images: input_images})

    def generation_step(self, sess, z_samples):
        """Generates new images"""
        return sess.run(self.generator, feed_dict={self.z_samples: z_samples})

    def recognition_step(self, sess, input_images):
        """Reconstruct images"""
        return sess.run(self.generated_images_sigmoid, feed_dict={self.input_images: input_images})


if __name__ == '__main__':
    # Let's test it before use
    cvae = ConvVAE(2, batch_size=1)
    init = tf.global_variables_initializer()

    z_sample = np.random.normal(size=2)

    print('z= {}'.format(z_sample))

    with tf.Session() as sess:
        sess.run(init)
        output_frame = cvae.generation_step(sess, np.reshape(z_sample, [1, 2]))
        output_frame = output_frame * 255
        output_frame = output_frame.astype(np.uint8)
        print('Shape= ', output_frame.shape)
        plt.imshow(np.reshape(output_frame, [144, 256, 3]))
        plt.show()
