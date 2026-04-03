import tensorflow as tf
from tensorflow import keras as keras
import os
import sys
import re
from image_dataset import create_dataset
from evaluate_image import evaluate_image
import numpy as np
from settings import settings
# Check if a GPU is available
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print(f"TensorFlow is using the following GPU(s): {gpu_devices}")
else:
    print("TensorFlow is using the CPU.")

print("Tensorflow: ", tf.version.VERSION)
print("Keras: ", keras.__version__)

from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Input
from keras.layers import Flatten, Dense, Reshape, BatchNormalization
from keras.models import Model
from keras import losses


class SimpleAutoencoder(Model):
    """
    Definition of the Example autoencoder model as a subclass of the TensorFlow Model class
    """
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
    """
    Definition of the Autoencoder model as a subclass of the TensorFlow Model class
    """
    def __init__(self, latent_dimensions, input_shape, loss, learning_rate=0.001):
        super(ConvAutoencoder, self).__init__()
        self.latent_dimensions = latent_dimensions
        self.shape = input_shape
        print("Input shape", input_shape, loss)
        # Building the encoder of the Auto-encoder
        self.encoder = tf.keras.Sequential([
            Input(shape=input_shape),  # Explicit Input layer
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            BatchNormalization(),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            MaxPooling2D((2, 2), padding='same'),
            BatchNormalization(),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2), padding='same')])
        self.encoder.summary()
        # Building the decoder of the Auto-encoder
        self.decoder = tf.keras.Sequential([
            Input(shape=self.encoder.output_shape[1:]),  # Explicit Input layer
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D((2, 2), interpolation='bilinear'),
            Conv2D(8, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D((2, 2), interpolation='bilinear'),
            Conv2D(16, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            UpSampling2D((2, 2), interpolation='bilinear'),
            Conv2D(1, (3, 3), activation='sigmoid', padding='same')])
        self.decoder.summary()
        if loss == 'binary_crossentropy':
            optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)
            self.compile(optimizer=optimizer, loss='binary_crossentropy')
        elif loss == 'mean_squared_error':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            self.compile(optimizer=optimizer, loss=losses.MeanSquaredError())
        else:
            self.compile()

    def call(self, input_data):
        self.encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(self.encoded_data)
        return decoded_data

    def init_weights(self, block_size: int, inpaint_size: int):
        checkpoint_path = "./weights_" + \
            str(block_size) + "_" + str(inpaint_size) + "/cp-{epoch:04d}.weights.h5"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        self.build(input_shape=(None,) + self.shape)
        return checkpoint_path, 0

    def load_weights(self, block_size: int, inpaint_size: int, inference_only: bool):
        checkpoint_path = "./weights_" + \
            str(block_size) + "_" + str(inpaint_size) + "/cp-{epoch:04d}.weights.h5"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        self.build(input_shape=(None,) + self.shape) 
        initial_epoch = 0
        if latest is not None:
            print("Loading weights from " + latest + "...")
            status = super().load_weights(latest)
#            super().save_weights(latest[:-4] + "h5")
            if inference_only:
                status.expect_partial()
            epoch_str = re.search(r'cp-\d+', latest)
            if epoch_str:
                initial_epoch = int(epoch_str.group()[3:])
        else:
            print("Warning: Loading weights from " + checkpoint_dir + " failed")
        return checkpoint_path, initial_epoch


def make_model(input_size=28, loss='', learning_rate=0.001):
    latent_dimensions = 64
    input_shape = (input_size, input_size, 1)
    return ConvAutoencoder(latent_dimensions, input_shape, loss, learning_rate)


def lr_drop_scheduler(epoch, learning_rate):
    drop_rate = 0.8
    epochs_drop = 5
    if epoch // epochs_drop > 0:
        return learning_rate
    else:
        return learning_rate * drop_rate


if __name__ == "__main__":
    # Create an instance of the Autoencoder model
    autoencoder = make_model(
        settings.input_size, settings.loss, settings.learning_rate)
    data_path = "./images"
    train, test, train_steps, validate_steps = create_dataset(
        data_path, settings.batch_size, settings.inpaint_size, autoencoder.shape)

    if len(sys.argv) > 1 and sys.argv[1] == 'init':
        train_epoch_count = 1
        checkpoint_path, initial_epoch = autoencoder.init_weights(
            settings.input_size, settings.inpaint_size)
    else:
        train_epoch_count = int(sys.argv[1]) if len(sys.argv) > 1 else 0
        checkpoint_path, initial_epoch = autoencoder.load_weights(
            settings.input_size, settings.inpaint_size, inference_only=False)

    save_freq = 'epoch'
    print("initial epoch:", initial_epoch, "batches#:", train_steps)
    use_tensorboard = len(sys.argv) > 2 and sys.argv[2] == "TensorBoard"

    if train_epoch_count > 0:
        callbacks = []
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, verbose=1, save_freq=save_freq)
        callbacks.append(cp_callback)

        if use_tensorboard:
            tb_callback = TensorBoard(log_dir='./log')
            callbacks.append(tb_callback)

        lr_scheduler_callback = LearningRateScheduler(lr_drop_scheduler)
        callbacks.append(lr_scheduler_callback)
        version_parts = tf.__version__.split('.')
        if float(version_parts[0] + '.' + version_parts[1] ) >= 2.16:
            autoencoder.fit(train, epochs=initial_epoch + train_epoch_count, initial_epoch=initial_epoch,
                        batch_size=settings.batch_size,
                        steps_per_epoch=train_steps,
                        shuffle=False,
                        validation_data=test,
                        validation_steps=validate_steps,
                        callbacks=callbacks)
        else:
            autoencoder.fit(train, epochs=initial_epoch + train_epoch_count, initial_epoch=initial_epoch,
                        batch_size=settings.batch_size,
                        steps_per_epoch=train_steps,
                        shuffle=False,
                        validation_data=test,
                        validation_steps=validate_steps,
                        use_multiprocessing=True,
                        workers=1,
                        callbacks=callbacks)
 
    else:
        autoencoder.build((settings.batch_size,) + autoencoder.shape)

    autoencoder.summary()
    # preview test
    files = tf.data.Dataset.list_files(data_path + "/test/*.JPG")
    out_path = "./output"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for image_path in files:
        result_image, original_image = evaluate_image(
            autoencoder, image_path, settings.inpaint_size, settings.max_dim, autoencoder.shape)
        base = os.path.basename(image_path.numpy().decode())
        tf.io.write_file(os.path.join(out_path, base + "_result.png"),
                         tf.io.encode_png(result_image.astype(np.uint8)))
        tf.io.write_file(os.path.join(out_path, base + "_original.png"),
                         tf.io.encode_png(original_image.astype(np.uint8)))
