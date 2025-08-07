import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

# Establish the directory from which images are pulled

data_dir = pathlib.Path('/home/dayseb/.keras/datasets/sebastian_pictures')

# Get amount of images in directory

image_count = len(list(data_dir.glob('*/*.jpg')))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split = 0.2,
  subset = "training",
  seed = 123
  image_size = (img_height, img_width),
  batch_size = batch_size)
