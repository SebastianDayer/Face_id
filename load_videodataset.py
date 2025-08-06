import tqdm
import random
import pathlib
import itertools
import collections

import os
import cv2
import numpy as np
import remotezip as rz

import tensorflow as tf

# modules for displaying an animation using imageio
import imageio
from IPython import display
from urllib import request
from tensorflow_docs.vis import embed


# The UCF101 dataset contains 101 categories of different actions in video, primarily used in action recognition. A subset of these categories is used in this 
# demo

URL = 'https://storage.googleapis.com/thumos14_files/UCF101_videos.zip'

# The above URL contains a zip file with the UCF 101 dataset. Create a function that uses the remotezip library to examine
# the contents of the zip file in that URL:

def list_files_from_zip_url(zip_url):
    """List the files in each class of the dataset given a URL with the zip file.
    
      Args: 
        zip_url: A URL from which the files can be extracted from.
        
      Returns:
        List of files in each of the classes.
    """
    files = []
    with rz.RemoteZip(zip_url) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

files = list_files_from_zip_url(URL)
files = [f for f in files if f.endswith('.avi')]
files[:10]

# Define get_class function that retrieves the class name from a filename. Then, create a function called get_files