import tensorflow as tf

print("TensorFlow version: ", tf.__version__)


# Loading and preparing MNIST dataset. Pixel values of images range from 0 through 255. Scaled these values to a range of 0 to 1 by dividing values by 255.0.
# This also converts the sample data from integers to floating-point numbers.  

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Building a tf.keras.Sequential model

model = tf.keras.models.Seqiential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
]) 

# model returns a vector of logits or log-odds scores, one for each class, this is saved to predictions 

predictions = model(x_train[:1]).numpy()

# The tf.nn.softmax function converts these logits to probabilities for each class

tf.nn.softmax(predictions).numpy() 

# Define a loss function for training using losses.SparseCategoricalCrossentropy:

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# The loss functions takes a vector of ground truth values and a vector of logits and returns a scalar loss for each example.
# This loss is equal to the negative log probability of the true class: The loss is zero if the model is sure of the correct class.

#This untrained model gives probability closs to random (1/10 for each class), so the initial loss should be close to -tf.math.log(1/10) ~= 2.3

loss_fn(y_train[:1], predictions).numpy()

# Before training, must configure and compile the model using Keras Model.compile. Set the optimizer class to adam, set the loss to the loss_fn function,
# and specify a metric to be evaluated for the model by setting the metrics parameter to accuracy
 
model.compile(optimizer = 'adam',
              loss = loss_fn,
              metrics = ['accuracy'])


# Use model.fit method to adjust your model parameters and minimize loss:
model.fit(x_train, y_train, epochs = 5)

# model.evaluate method checks the model's performance, usually on a validation or test set
model.evaluate(x_test, y_test, verbose=2)

