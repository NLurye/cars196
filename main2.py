import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np

# Random seed
tf.random.set_seed(1)

# Hyper-parameters
batch_size = 32
epochs = 40
learning_rate = 0.001
n_workers = 8

# Download data
cars_test, cars_val, cars_train = tfds.load('Cars196', data_dir='C:/Users/anast/PycharmProjects/cars196/data',
                                            as_supervised=False,
                                            shuffle_files=True, split=["test", "train[0%:20%]", "train[20%:]"])


# Iterate over the dataset and crop the images using the bounding box information
def crop_images(example):
    image = example['image']
    bbox = example['bbox']
    height, width = tf.unstack(tf.shape(image)[:2])
    scaled_box = bbox * [height, width, height, width]
    ymin, xmin, ymax, xmax = tf.unstack(tf.cast(scaled_box, tf.int32))
    box_width = xmax - xmin
    box_height = ymax - ymin
    image = tf.image.crop_to_bounding_box(image, ymin, xmin, box_height, box_width)

    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, 224, 224)
    image = preprocess_input(image)
    label = tf.one_hot(example['label'], 196)

    return image, label

cars_train = cars_train.map(crop_images, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
cars_val = cars_val.map(crop_images, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

# Dictionary of the labels - maps between the label (int) number and the  vehicle model (str)
label_dic = pd.read_csv('/home/anastasia/Downloads/labels_dic.csv', header=None, dtype={0: str}).\
    set_index(0).squeeze().to_dict()

# Create model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom prediction layer
x = base_model.output
x = tf.keras.layers.GlobalMaxPooling2D()(x)  #GlobalAveragePooling2D
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)

# Combine base model and prediction layer
prediction_layer = tf.keras.layers.Dense(196, activation='softmax')(x)
output = prediction_layer
model = Model(inputs=base_model.input, outputs=output)
model.summary()

# Train model
opt = Adam(lr=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
hist = model.fit(cars_train, epochs=epochs, batch_size=batch_size,
                 validation_data=cars_val, workers=n_workers)
model.save('model')

# Analyze results
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
plt.show()