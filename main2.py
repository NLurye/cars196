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
n_workers = 1

# Use ImageDataGenerator to apply data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)


# Download data
cars_test, cars_val, cars_train = tfds.load('Cars196', data_dir='C:/Users/anast/PycharmProjects/cars196/data',
                                            as_supervised=False,
                                            shuffle_files=True, split=["test", "train[0%:20%]", "train[20%:]"])

from keras.utils import load_img, img_to_array
# Iterate over the dataset and crop the images using the bounding box information
def preprocess_image(example):
    image = example['image']
    print('before: ', image.shape)
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
    print('after: ', image.shape)

    return image, label
def tfds_imgen(ds, imgen, batch_size, batches_per):
    for images, labels in ds:
        flow_ = imgen.flow(images, labels, batch_size=batch_size)
        for _ in range(batches_per):
            yield next(flow_)
# for example in cars_train:
#     image, label = example['image'], example['label']
#     train_datagen.fit(np.array([image]))
# # datagen.flow(x_train,y_train, batch_size=6)
# train_datagen.fit(cars_train)


cars_train_pp = cars_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
cars_val_pp = cars_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

cars_train_pp = tfds_imgen(
    cars_train_pp.as_numpy_iterator(), train_datagen,
    batch_size=32, batches_per= 32)

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
hist = model.fit(cars_train_pp, epochs=epochs, validation_data=cars_val_pp, workers=n_workers)
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