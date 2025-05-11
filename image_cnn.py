import numpy as np
import math
import pandas as pd
import tensorflow as tf
import os
from keras.callbacks import TensorBoard
import re 
from image_dataset import process_path, create_dataset
import cv2 as cv
from evaluate_image import evaluate_image
# Check if a GPU is available
gpu_devices = tf.config.list_physical_devices('GPU')

if not gpu_devices:
    print("TensorFlow is using the CPU.")
else:
    print(f"TensorFlow is using the following GPU(s): {gpu_devices}")

print("Tensorflow: ", tf.version.VERSION)
print("Keras: ", tf.keras.__version__)

from keras import layers, losses
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D 

# Definition of the Autoencoder model as a subclass of the TensorFlow Model class
class SimpleAutoencoder(Model):
    def __init__(self, latent_dimensions, data_shape):
        super(SimpleAutoencoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.data_shape = data_shape

        # Encoder architecture using a Sequential model
        self.encoder = tf.keras.Sequential([
            Flatten(),
            Dense(latent_dimensions, activation='relu'),
        ])

        # Decoder architecture using another Sequential model
        self.decoder = tf.keras.Sequential([
            Dense(tf.math.reduce_prod(data_shape), activation='sigmoid'),
            Reshape(data_shape)
        ])
        self.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Forward pass method defining the encoding and decoding steps
    def call(self, input_data):
        encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

class ConvAutoencoder(Model):
    def __init__(self, latent_dimensions, input_shape, loss = 'binary_crossentropy'):
        super(ConvAutoencoder, self).__init__(input_shape)
        self.latent_dimensions = latent_dimensions
        print("Input shape", input_shape, loss)
        # Building the encoder of the Auto-encoder
        self.encoder = tf.keras.Sequential([
             Conv2D(16, (3, 3), activation ='relu', padding ='same', input_shape=input_shape),
             MaxPooling2D((2, 2), padding ='same'),
             Conv2D(8, (3, 3), activation ='relu', padding ='same'),
             MaxPooling2D((2, 2), padding ='same'),
             Conv2D(8, (3, 3), activation ='relu', padding ='same'),
             MaxPooling2D((2, 2), padding ='same')])

                # Building the decoder of the Auto-encoder 
        self.decoder = tf.keras.Sequential([
             Conv2D(8, (3, 3), activation ='relu', padding ='same'),
             UpSampling2D((2, 2)),
             Conv2D(8, (3, 3), activation ='relu', padding ='same'),
             UpSampling2D((2, 2)),
             Conv2D(16, (3, 3), activation ='relu'),
             UpSampling2D((2, 2)),
             Conv2D(1, (3, 3), activation ='sigmoid', padding ='same')])
        if loss =='binary_crossentropy':
             self.compile(optimizer ='adadelta', loss ='binary_crossentropy')
        else: # mean_squared
             self.compile(optimizer='adam', loss=losses.MeanSquaredError())

    # Forward pass method defining the encoding and decoding steps
    def call(self, input_data):
        self.encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(self.encoded_data)
        return decoded_data

def make_model(loss='binary_crossentropy'):
    latent_dimensions = 64
    input_shape = (28,28,1) 
    autoencoder = ConvAutoencoder(latent_dimensions, input_shape, loss)
    return autoencoder, input_shape


if __name__ == "__main__":
    # Specifying the dimensionality of the latent space and training parameters
    batch_size=64
    inpaint_size = 8
    max_dim = 640
    data_path = "./images"
    mode = "latent"
    train_epoch_count = 1 # if 0 then test mode only
    if inpaint_size > 0:
        mode = "inpaint"
    loss = "mean_squared_error" # can be 'binary_crossentropy'
    # Creating an instance of the Autoencoder model
    autoencoder, input_shape = make_model(loss)
    train, test = create_dataset(data_path, batch_size, inpaint_size, input_shape)

    n_batches_per_epoch = train.cardinality() // batch_size
    print("batches#:", n_batches_per_epoch)
    print(input_shape)
    print(train.element_spec)
    print(test.element_spec)

    checkpoint_path = "training_" + mode + "/cp-{epoch:04d}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)

    initial_epoch = 0
    if latest is not None:
       print("Loading weights from " +  latest + "...")
       autoencoder.load_weights(latest)
       str_list = re.findall(r'\d+', latest)
       if len(str_list) == 1:
         initial_epoch = int(str_list[-1])

    #save_freq=5*n_batches
    save_freq = 'epoch'
    epochs=initial_epoch + train_epoch_count

    if epochs>initial_epoch:
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=save_freq)
        tb_callback = TensorBoard(log_dir ='./log')

        autoencoder.fit(train, epochs=epochs, initial_epoch=initial_epoch,
                                batch_size=None,
                shuffle=False,
                                workers=4,
                validation_data=test,
                                use_multiprocessing=True,
                                callbacks=[cp_callback, tb_callback])
    else:
       autoencoder.build((batch_size,) + input_shape)

    autoencoder.summary()

    files = tf.data.Dataset.list_files(data_path + "/test/*.jpg")
    for image_path in files:
       evaluate_image(autoencoder, image_path, inpaint_size, max_dim, input_shape)
