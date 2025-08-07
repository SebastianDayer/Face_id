import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib

# Establish the directory from which images are pulled

image_dir = pathlib.Path('/home/dayseb/.keras/datasets/face_detection_photos')
data_dir = image_dir.glob('*/*.jpg')

# Get amount of images in directory

image_count = len(list(data_dir))

print(image_count)