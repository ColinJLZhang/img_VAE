#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-01 15:18:38
# @Author  : mrobotor (colinzhang@applecore.cc)
# @Link    : http://darklunar.ml
# @Version : $Id$

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from conv_vae import ConvVAE
from data_utils import read_dataset
import time
from scipy.stats import norm

# data processing
# read data set
train_ds, valid_ds = read_dataset('./imgdata', test_size = 0.097)
print(train_ds.images().shape)
print((train_ds.images().nbytes + valid_ds.images().nbytes) / (1024.0 * 1024.0), 'MB')

latent_dim = 10
batch_size = 50

# let's create ConvVAE
cvae = ConvVAE(latent_dim, batch_size)

# let's train ConvVAE
num_epochs = 15
interval = 200

saver = tf.train.Saver(max_to_keep = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    t = time.time()
    # for num of epochs    
    while(train_ds.epochs_completed() < num_epochs):
        
        current_epoch = train_ds.epochs_completed()
        step = 0
        print('[----Epoch {} is started ----]'.format(current_epoch))
        
        # take next batch until epoch is completed        
        while(train_ds.epochs_completed() < current_epoch + 1):
            input_images = train_ds.next_batch(batch_size)
            # do training step
            cvae.training_step(sess, input_images)
            step += 1
            
            if step % interval == 0:
                print('loss: {} validation loss: {}'.format(cvae.loss_step(sess, input_images),\
                                                            cvae.loss_step(sess, valid_ds.next_batch(batch_size))))
                
        print('[----Epoch {} is finished----]'.format(current_epoch))
        saver.save(sess, './checkpoints/', global_step=current_epoch)
        # print('[----Checkpoint is saved----]')
     
    print('Training time: {}s'.format(time.time() - t))
    
    # let's see how well our model reconstructs input images       
    input_images = train_ds.next_batch(batch_size)

    output_images = cvae.recognition_step(sess, input_images)
    output_images = output_images * 255
    output_images = output_images.astype(np.uint8)
    print('Shape= ', output_images.shape)


# Let's plot them!!!
w = 10
h = 5
figure = np.zeros([144 * h, 256 * w, 3], dtype = np.uint8)
k = 0
for i in range(h):
    for j in range(w):
        image = np.reshape(output_images[k], [144, 256, 3])
        figure[i * 144: (i + 1) * 144,
               j * 256: (j + 1) * 256,
               :] = image
        k += 1
    
plt.figure(figsize=(15, 15))


# generate a new video frame
# reset computational graph
tf.reset_default_graph()

# create model
cvae = ConvVAE(latent_dim, batch_size = 1)

# restoration
saver = tf.train.Saver()
path = tf.train.latest_checkpoint('./checkpoints')

with tf.Session() as sess:
    # restore session    
    saver.restore(sess, path)
    
    # let's create random latent vector from normal distribution     
    # z = np.random.normal(size = latent_dim)
    
    vals = np.array([np.linspace(0.05, 0.95, latent_dim, dtype = np.float32) for i in range(latent_dim)])
    vals = np.reshape(vals, [latent_dim * latent_dim])
    z_samples = norm.ppf(vals)
    z_samples = np.random.permutation(z_samples)
            
    print('z=',z_samples[:latent_dim])
    
    # Generate a new video frame
    output_image = cvae.generation_step(sess, np.reshape(z_samples[:latent_dim], [1, latent_dim]))
    output_image = output_image * 255
    output_image = output_image.astype(np.uint8)
    
    print('Shape=',output_image.shape)
    
    # plot it
    plt.imshow(np.reshape(output_image, [64, 64, 3]))
    plt.show()