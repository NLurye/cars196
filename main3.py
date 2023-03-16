# END TO END CNN
import tensorflow as tf
import keras
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input, layers
import pandas as pd
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, GlobalAveragePooling2D

# Random seed
tf.random.set_seed(1)

# Hyper-parameters
batch_size = 32
epochs = 50
learning_rate = 0.0001
input_shape = (224, 224, 3)
num_classes = 196
loss_fn = keras.losses.CategoricalCrossentropy()
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
data_standartization = tf.keras.Sequential([
    layers.Lambda(lambda x: preprocess_input(x))], name='data_standartization_16')
# n_workers = 4

# Download data
cars = tfds.load('Cars196', as_supervised=False, shuffle_files=True)

cars_dataset = cars['train'].concatenate(cars['test'])

# Split the dataset into 10/90 train-test split
train_size = int(0.8 * len(cars_dataset))
test_size = len(cars_dataset) - train_size
cars_train = cars_dataset.take(train_size)
cars_test = cars_dataset.skip(train_size)

# Split the train dataset into 10/90 validation-train split
valid_size = int(0.1 * len(cars_train))
train_size = len(cars_train) - valid_size
cars_val = cars_train.take(valid_size)
cars_train = cars_train.skip(valid_size)

print(f'Dataset size: {len(cars_dataset)} images')
print(f'Test size: {len(cars_test)} images')
print(f'Val size: {len(cars_val)} images')
print(f'Train size: {len(cars_train)} images')

# Dictionary of the labels - maps between the label (int) number and the  vehicle model (str)
label_dic = pd.read_csv('labels_dic.csv', header=None, dtype={0: str}).set_index(0).squeeze().to_dict()


# Iterate over the dataset and crop the images using the bounding box information
def preprocess_image(example):
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
    label = tf.one_hot(example['label'], num_classes)
    print('after: ', image.shape)

    return image, label


data_augmentation = tf.keras.Sequential([layers.RandomFlip("horizontal"), layers.RandomRotation(0.1)])

data_augmentation_cnn = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2, 0.2),
    layers.RandomTranslation(0.2, 0.2),
], name='data_augmentation_cnn')

# cars_train_pp = cars_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
# cars_train_pp = cars_train_pp.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
# cars_train_pp = cars_train_pp.batch(batch_size)
# cars_val_pp = cars_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

cars_train_pp = cars_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
    buffer_size=tf.data.AUTOTUNE)
cars_train_pp_aug = cars_train_pp.map(lambda x, y: (data_augmentation_cnn(x, training=True), y),
                                      num_parallel_calls=tf.data.AUTOTUNE)
cars_val_pp = cars_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
    buffer_size=tf.data.AUTOTUNE)
cars_test_pp = cars_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size).prefetch(
    buffer_size=tf.data.AUTOTUNE)

opt = Adam(learning_rate=learning_rate)

model_cnn_1 = keras.models.Sequential([tf.keras.Input(shape=input_shape), data_standartization])

model_cnn_1.add(Conv2D(filters=16, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))

model_cnn_1.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))
model_cnn_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn_1.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))
model_cnn_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn_1.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))
model_cnn_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn_1.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))
model_cnn_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn_1.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))
model_cnn_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn_1.add(Conv2D(filters=1024, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))
model_cnn_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn_1.add(Conv2D(filters=2048, kernel_size=(3, 3), padding="same"))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation("relu"))
model_cnn_1.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_cnn_1.add(GlobalAveragePooling2D())

model_cnn_1.add(Dense(4096, use_bias=False))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation('relu'))

model_cnn_1.add(Dense(2048, use_bias=False))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation('relu'))

model_cnn_1.add(Dense(1024, use_bias=False))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation('relu'))

model_cnn_1.add(Dense(512, use_bias=False))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation('relu'))

model_cnn_1.add(Dense(256, use_bias=False))
model_cnn_1.add(BatchNormalization())
model_cnn_1.add(Activation('relu'))

model_cnn_1.add(Dense(num_classes, activation='softmax'))

model_cnn_1.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

# Train the model
history1 = model_cnn_1.fit(cars_train_pp, epochs=epochs, validation_data=cars_val_pp, callbacks=[early_stopping])

history2 = model_cnn_1.fit(cars_train_pp_aug, epochs=epochs, validation_data=cars_val_pp, callbacks=[early_stopping])

model_cnn_1.save('model_cnn_1')


hist = {}
for key in history1.history:
    hist[key] = history1.history[key] + history2.history[key]

plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('New_Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

opt = Adam(learning_rate=learning_rate)

model_2 = Sequential()

model_2.add(Conv2D(input_shape=input_shape, filters=16, kernel_size=(3, 3), padding="same"))
model_2.add(BatchNormalization())
model_2.add(Activation("relu"))

model_2.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same"))
model_2.add(BatchNormalization())
model_2.add(Activation("relu"))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_2.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same"))
model_2.add(BatchNormalization())
model_2.add(Activation("relu"))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_2.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same"))
model_2.add(BatchNormalization())
model_2.add(Activation("relu"))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_2.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same"))
model_2.add(BatchNormalization())
model_2.add(Activation("relu"))
model_2.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model_2.add(GlobalAveragePooling2D())
model_2.add(Flatten())

model_2.add(Dense(256, activation='relu', use_bias=False))
model_2.add(Dense(512, activation='relu', use_bias=False))

model_2.add(Dense(num_classes, activation='softmax'))

# compile the model
model_2.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])

# Train the model
history1 = model_2.fit(cars_train_pp, epochs=epochs, validation_data=cars_val_pp, callbacks=[early_stopping])

history2 = model_2.fit(cars_train_pp_aug, epochs=epochs, validation_data=cars_val_pp, callbacks=[early_stopping])
model_2.save('model_2')

hist = {}
for key in history1.history:
    hist[key] = history1.history[key] + history2.history[key]

plt.plot(hist['loss'])
plt.plot(hist['val_loss'])
plt.title('New_Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

import numpy as np
from sklearn.metrics import classification_report

target_names = [label_dic[str(i)] for i in range(num_classes)]

accuracy_1 = model_cnn_1.evaluate(cars_test_pp)[1]
print("First model, accuracy: {:5.2f}%".format(100 * accuracy_1))

y_pred = model_cnn_1.predict(cars_test_pp)
y_true = np.concatenate([y.numpy() for x, y in cars_test_pp], axis=0)
y_true = np.argmax(y_true, axis=1)
print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=target_names))

accuracy_1 = model_2.evaluate(cars_test_pp)[1]
print("First model, accuracy: {:5.2f}%".format(100 * accuracy_1))

y_pred = model_2.predict(cars_test_pp)
y_true = np.concatenate([y.numpy() for x, y in cars_test_pp], axis=0)
y_true = np.argmax(y_true, axis=1)
print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=target_names))
