import tensorflow as tf
from tensorflow import keras as keras
import os
import sys
import re
from image_dataset import create_dataset
from evaluate_image import evaluate_image
import numpy as np
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
from keras.callbacks import TensorBoard
from keras.callbacks import LearningRateScheduler


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
    def __init__(self, latent_dimensions, input_shape, loss, learning_rate=0.001):
        super(ConvAutoencoder, self).__init__(input_shape)

        print("Input shape", input_shape, loss)
        # Building the encoder of the Auto-encoder
        self.encoder = tf.keras.Sequential([
            Conv2D(16, (3, 3), activation ='relu', padding ='same', input_shape=input_shape),
            MaxPooling2D((2, 2), padding ='same'),
            Conv2D(8, (3, 3), activation ='relu', padding ='same'),
            MaxPooling2D((2, 2), padding ='same'),
            Conv2D(8, (3, 3), activation ='relu', padding ='same'),
            MaxPooling2D((2, 2), padding ='same')])
        self.encoder.summary()
        # Building the decoder of the Auto-encoder
        self.decoder = tf.keras.Sequential([
            Conv2D(8, (3, 3), activation ='relu', padding ='same', input_shape=self.encoder.output_shape[1:] ),
            UpSampling2D((2, 2), interpolation='bilinear'),
            Conv2D(8, (3, 3), activation ='relu', padding ='same'),
            UpSampling2D((2, 2), interpolation='bilinear'),
            Conv2D(16, (3, 3), activation ='relu', padding ='same'),
            UpSampling2D((2, 2), interpolation='bilinear'),
            Conv2D(1, (3, 3), activation ='sigmoid', padding ='same')])
        self.decoder.summary()

        if loss =='binary_crossentropy':
            optimizer = keras.optimizers.Adadelta(learning_rate=learning_rate)
            self.compile(optimizer=optimizer, loss ='binary_crossentropy')
        elif loss == 'mean_squared_error':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            self.compile(optimizer=optimizer, loss=losses.MeanSquaredError())
        else:
            self.compile()


    # Forward pass method defining the encoding and decoding steps
    def call(self, input_data):
        self.encoded_data = self.encoder(input_data)
        decoded_data = self.decoder(self.encoded_data)
        return decoded_data


    def load_weights(self, block_size: int, inpaint_size: int, inference_only: bool):
        checkpoint_path = "./weights_" + str(block_size) + "_" + str(inpaint_size) + "/cp-{epoch:04d}.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)

        initial_epoch = 0
        if latest is not None:
            print("Loading weights from " + latest + "...")
            status = super().load_weights(latest)
            if inference_only:
                status.expect_partial()
            str_list = re.findall(r'\d+', latest)
            if len(str_list) > 0:
                initial_epoch = int(str_list[-1])
        else:
            print("Warning: Loading weights from " + checkpoint_dir + " failed")
        return checkpoint_path, initial_epoch


def make_model(input_size=28, loss='', learning_rate=0.001):
    latent_dimensions = 64
    input_shape = (input_size, input_size, 1)
    autoencoder = ConvAutoencoder(latent_dimensions, input_shape, loss, learning_rate)
    return autoencoder, input_shape


class settings():
    batch_size = 64
    inpaint_size = 0
    input_size = 24
    # max processing size, if an input image has bigger dims it is resized for faster processing
    max_dim = 640
    # grayscale images only
    channels = 1
    # use low-variance block mask
    zero_diff = 0
    #loss can be 'binary_crossentropy', "mean_squared_error" or empty string for inference mode
    loss = 'mean_squared_error'
    learning_rate=0.001


def lr_drop_scheduler(epoch, learning_rate):
    drop_rate = 0.8
    epochs_drop = 5
    if epoch // epochs_drop > 0:
        return learning_rate
    else:
        return learning_rate * drop_rate


if __name__ == "__main__":
    # Create an instance of the Autoencoder model
    autoencoder, input_shape = make_model(settings.input_size, settings.loss, settings.learning_rate)
    data_path = "./images"
    train, test = create_dataset(data_path, settings.batch_size, settings.inpaint_size, input_shape)
    checkpoint_path, initial_epoch = autoencoder.load_weights(settings.input_size, settings.inpaint_size, inference_only = False)
    save_freq = 'epoch'
    n_batches_per_epoch = -1
    if train.cardinality() != tf.data.INFINITE_CARDINALITY and train.cardinality() != tf.data.UNKNOWN_CARDINALITY:
        n_batches_per_epoch = int(train.cardinality()) // settings.batch_size

    print("initial epoch:", initial_epoch, "batches#:", n_batches_per_epoch)
    train_epoch_count = train_epoch_count = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    use_tensorboard = len(sys.argv) > 2 and sys.argv[2] =="TensorBoard"

    if train_epoch_count>0:
        callbacks = []
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
            save_weights_only=True, verbose=1, save_freq=save_freq)
        callbacks.append(cp_callback)

        if use_tensorboard:
            tb_callback = TensorBoard(log_dir ='./log')
            callbacks.append(tb_callback)

        lr_scheduler_callback = LearningRateScheduler(lr_drop_scheduler)
        callbacks.append(lr_scheduler_callback)

        autoencoder.fit(train, epochs=initial_epoch + train_epoch_count, initial_epoch=initial_epoch,
            batch_size=settings.batch_size,
            shuffle=False,
            workers=4,
            validation_data=test,
            use_multiprocessing=True,
            callbacks=callbacks)
    else:
        autoencoder.build((settings.batch_size,) + input_shape)

    autoencoder.summary()
    # preview test
    files = tf.data.Dataset.list_files(data_path + "/test/*.jpg")
    out_path = "./output"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for image_path in files:
        result_image, original_image = evaluate_image(autoencoder, image_path, settings.inpaint_size, settings.max_dim, input_shape)
        base = os.path.basename(image_path.numpy().decode())
        tf.io.write_file(os.path.join(out_path, base + "_result.png"),
                            tf.io.encode_png(result_image.astype(np.uint8)))
        tf.io.write_file(os.path.join(out_path, base + "_original.png"),
                            tf.io.encode_png(original_image.astype(np.uint8)))

