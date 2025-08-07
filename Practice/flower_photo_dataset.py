import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

import pathlib 
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path('/home/dayseb/.keras/datasets/flower_photos.tgz/flower_photos')

image_count = len(list(data_dir.glob('*/*.jpg')))

roses = list(data_dir.glob('roses/*'))

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "training",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.2,
    subset = "validation",
    seed = 123,
    image_size = (img_height, img_width),
    batch_size = batch_size,)

class_names = train_ds.class_names

import matplotlib.pyplot as plt

plt.figure(figsize = (10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

# for image_batch, labels_batch in train_ds:
    # print(image_batch.shape)
    # print(labels_batch.shape)

# normalization_layer = tf.keras.layers.Rescaling(1./255)

# normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
# image_batch, labels_batch = next(iter(normalized_ds))
# first_image = image_batch[0]

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation = 'relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = 'relu'),
    tf.keras.layers.Dense(num_classes)
])

model.compile(
    optimizer = 'adam',
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = ['accuracy']
)

model.fit(
    train_ds,
    validation_data = val_ds,
    epochs = 3
)

# list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'), shuffle = False)
# list_ds = list_ds.shuffle(image_count, reshuffle_each_iteration = False)

# class_names = np.array(sorted([item.name for item in data_dir.glob('*') if item.name != "LICENSE.txt"]))

# val_size = int(image_count * 0.2)
# train_ds = list_ds.skip(val_size)
# val_ds = list_ds.take(val_size)

# def get_label(file_path):
#     # Convert path to a list of path components
#     parts = tf.strings.split(file_path, os.path.sep)
#     # The second to last is the class-directory
#     one_hot = parts[-2] == class_names 
#     # Integer encode the label
#     return tf.argmax(one_hot)

# def decode_img(img):
#     # Convert the compressed string to a 3D uint8 tensor
#     img = tf.io.decode_jpeg(img, channels = 3)
#     # Resize the image to the desired size
#     return tf.image.resize(img, [img_height, img_width])

# def process_path(file_path):
#     label = get_label(file_path)
#     # Load the raw data from the file as a string
#     img = tf.io.read_file(file_path)
#     img = decode_img(img)
#     return img, label

# # Set 'num_parallel_calls' so multiple images are loaded/processed in parallel.

# train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
# val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

# def configure_for_performance(ds):
#     ds = ds.cache()
#     ds = ds.shuffle(buffer_size = 1000)
#     ds = ds.batch(batch_size)
#     ds = ds.prefetch(buffer_size = AUTOTUNE)
#     return ds

# train_ds = configure_for_performance(train_ds)
# val_ds = configure_for_performance(val_ds)

# image_batch, label_batch = next(iter(train_ds))

# plt.figure(figsize = (10. 10))
# for i in range(9):
#     ax = plt.subplot(3, 3, i + 1)
#     plt.imshow(image_batch[i].numpy().astype('uint8'))
#     label = label_batch[i]
#     plt.title(class_names[label])
#     plt.axis('off')

