import os
import numpy as np
from random import choice
from keras_preprocessing.image import load_img, img_to_array

# Load in the data filenames
sketch_filenames = os.listdir("sketch")
color_filenames = os.listdir("color")

n = len(sketch_filenames)


def sketch_batch(batch_size):
    return _generate_batch(batch_size, "sketch")


def color_batch(batch_size):
    return _generate_batch(batch_size, "color")


def _generate_batch(batch_size, dataset):
    # Not sure if this is the most efficient way to do this
    batch = []
    for _ in range(batch_size):
        im = load_img(os.path.join(dataset, choice(sketch_filenames if dataset == "sketch" else color_filenames)),
                      color_mode="grayscale" if dataset == "sketch" else "rgb")
        batch.append(img_to_array(im))

    return np.array(batch)
