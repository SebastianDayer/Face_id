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