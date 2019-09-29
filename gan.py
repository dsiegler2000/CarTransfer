import os
import time
import traceback
from random import randrange
from keras.preprocessing.image import array_to_img
import cv2
import numpy as np
from keras import Input, Model, Sequential
import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, BatchNormalization, Activation, Deconvolution2D
from keras.optimizers import Adam
from squeezenet import SqueezeNet
import data_manager
from datetime import timedelta

"""
TODO:
Learning rate decay
Tuning the networks
More capable generator network
"""

tf.flags.DEFINE_integer("num_epochs", 1, "Number of epochs")
tf.flags.DEFINE_integer("batch_size", 8, "Batch size")
tf.flags.DEFINE_integer("input_width", 512, "Width of the input image")
tf.flags.DEFINE_integer("input_height", 512, "Height of the input image")
tf.flags.DEFINE_float("lr", 0.0002, "The learning rate for all optimizers")
tf.flags.DEFINE_float("beta_1", 0.6, "The beta_1 parameter for the Adam optimizer. Controls the moving average "
                                     "calculation. Should be set close to 1")

FLAGS = tf.flags.FLAGS

num_epochs = FLAGS.num_epochs
batch_size = FLAGS.batch_size
input_size = (FLAGS.input_width, FLAGS.input_height)
lr = FLAGS.lr
beta_1 = FLAGS.beta_1


def create_adam():
    return Adam(lr=lr, beta_1=beta_1)


def generator_conv(model, filters, name, kernel_size=(2, 2), stride=2):
    model.add(Conv2D(filters, kernel_size, strides=stride, name=f"{name}/conv_{kernel_size[0]}x{kernel_size[1]}"))
    model.add(BatchNormalization(name=f"{name}/bn"))
    model.add(Activation("relu", name=f"{name}/relu"))


def generator_deconv(model, filters, name, kernel_size=(2, 2), stride=2):
    model.add(Deconvolution2D(filters, kernel_size, strides=stride,
                              name=f"{name}/deconv_{kernel_size[0]}x{kernel_size[1]}"))
    model.add(BatchNormalization(name=f"{name}/bn"))
    model.add(Activation("relu", name=f"{name}/relu"))


def create_generator():
    """
    Creates the generator network
    :return: The generator network
    """
    # Following the model shown here: https://arxiv.org/pdf/1705.01908.pdf
    model = Sequential()
    generator_conv(model, 64, name="encoders/conv_A")
    generator_conv(model, 128, name="encoders/conv_B")
    generator_conv(model, 256, name="encoders/conv_C")
    generator_conv(model, 512, name="encoders/conv_D")
    generator_conv(model, 512, name="encoders/conv_E")
    generator_conv(model, 512, name="encoders/conv_F")
    generator_conv(model, 512, name="encoders/conv_G")
    generator_conv(model, 512, name="encoders/conv_transfer")

    generator_deconv(model, 512, name="decoder/deconv_G")
    generator_deconv(model, 512, name="decoder/deconv_F")
    generator_deconv(model, 512, name="decoder/deconv_E")
    generator_deconv(model, 512, name="decoder/deconv_D")
    generator_deconv(model, 256, name="decoder/deconv_C")
    generator_deconv(model, 128, name="decoder/deconv_B")
    generator_deconv(model, 64, name="decoder/deconv_A")

    model.add(Deconvolution2D(3, (2, 2), strides=2, name="output_conv"))

    model.compile(loss="binary_crossentropy", optimizer=create_adam())
    return model


def create_discriminator():
    """
    Makes the discriminator network
    :return: The discriminator network
    """
    # We are using the SqueezeNet because it is fairly resource-light and GANs can already be hard to train
    disc_net = SqueezeNet(input_width=input_size[0], classes=2)

    # Compile with loss
    disc_net.compile(loss="binary_crossentropy", optimizer=create_adam(), metrics=["accuracy"])

    return disc_net


def create_gan(d, g):
    """
    Makes the GAN
    :param d: The discriminator network
    :param g: The generator network
    :return: The GAN network
    """

    # Set this to false just for good practice
    d.trainable = False

    # Input placeholder
    gan_input = Input(shape=(input_size[0], input_size[1], 1))

    # Generate a color
    color = g(gan_input)

    # Generate a discriminator prediction
    pred = d(color)

    # Make and compile the model
    gan = Model(inputs=gan_input, outputs=pred)
    gan.compile(loss='binary_crossentropy', optimizer='adam')

    return gan


# Create the GAN
generator = create_generator()
discriminator = create_discriminator()

gan = create_gan(discriminator, generator)

# Train
start_time = time.time()

# Set up some logging stuff
log_filepath = os.path.join("logs", str(int(time.time())))
tensorboard_disc_filepath = os.path.join(log_filepath, "tensorboard_disc")
tensorboard_gan_filepath = os.path.join(log_filepath, "tensorboard_gan")


def write_log(callback, names, logs, batch):
    if not isinstance(logs, list):
        logs = [logs]
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch)
        callback.writer.flush()


try:
    if not os.path.exists(log_filepath):
        os.mkdir(log_filepath)
    if not os.path.exists(tensorboard_disc_filepath):
        os.mkdir(tensorboard_disc_filepath)
    if not os.path.exists(tensorboard_gan_filepath):
        os.mkdir(tensorboard_gan_filepath)

    tensorboard_disc_callback = TensorBoard(log_dir=tensorboard_disc_filepath)
    tensorboard_disc_callback.set_model(discriminator)
    tensorboard_gan_callback = TensorBoard(log_dir=tensorboard_gan_filepath)
    tensorboard_gan_callback.set_model(gan)

    for i in range(num_epochs):
        epoch_start_time = time.time()
        num_batches = int(data_manager.n / batch_size)
        for j in range(num_batches):
            if i != 0:
                epoch_rate = (time.time() - start_time) / i
                epoch_total_time = epoch_rate * num_epochs
                epoch_time_left = epoch_total_time - (time.time() - start_time)
                print(f"Estimated time remaining total: {timedelta(seconds=epoch_time_left)}")

            if j != 0:
                batch_rate = (time.time() - epoch_start_time) / j
                batch_total_time = batch_rate * num_batches
                batch_time_left = batch_total_time - (time.time() - epoch_start_time)
                print(f"Epoch {i} / {num_epochs}, batch {j} / {num_batches}, "
                      f"estimated time remaining for this epoch: {timedelta(seconds=batch_time_left)}")

            # Load a batch of sketches and colored images
            sketch_batch = data_manager.sketch_batch(batch_size)
            color_batch = data_manager.color_batch(batch_size)

            # Generate
            generated_images = generator.predict(sketch_batch)

            # Make the discriminator training set
            X = np.concatenate([color_batch, generated_images])
            # [1, 0] is real, [0, 1] is generated
            y_disc = np.array(batch_size * [[1, 0]] + batch_size * [[0, 1]])

            # Train discriminator
            discriminator.trainable = True
            logs = discriminator.train_on_batch(X, y_disc)
            write_log(tensorboard_disc_callback, ["train_loss", "disc_train_accuracy"], logs, i * num_batches + j)
            discriminator.trainable = False

            # Get a new sketch batch and label them all as 1 (consistent with above)
            sketch_batch = data_manager.sketch_batch(batch_size)
            y_gen = np.array(batch_size * [[1, 0]])

            # Train GAN batch
            logs = gan.train_on_batch(sketch_batch, y_gen)
            write_log(tensorboard_gan_callback, ["train_loss"], logs, i * num_batches + j)

            # 211/212
            cv2.imwrite(os.path.join(log_filepath, "generated_sample.png"),
                        generated_images[randrange(0, batch_size), :, :, :])
            generated_sample = array_to_img(generated_images[randrange(0, batch_size), :, :, :])
            generated_sample.save(os.path.join(log_filepath, "generated_sample.png"))

except KeyboardInterrupt as e:
    traceback.print_exc()
except Exception as e:
    traceback.print_exc()
finally:
    gan.save(os.path.join(log_filepath, "gan_model.h5"))
