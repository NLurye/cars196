import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.utils import load_img
from keras.preprocessing import image

# %% [code]
import keras
import tensorflow_datasets as tfds
from keras.optimizers import Adam
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras import layers
from keras.utils import np_utils
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import tensorflow_addons as tfa


cars_test, cars_val, cars_train = tfds.load('Cars196', data_dir='C:/Users/anast/PycharmProjects/cars196/data', as_supervised=False, shuffle_files=True,
                                            split=["test", "train[0%:20%]", "train[20%:]"])

label_dic = pd.read_csv('labels_dic.csv', header=None, dtype={0: str}).set_index(0).squeeze().to_dict()


def plot_single_example(data_iterator, label_dic=label_dic):
    image = data_iterator['image']
    label = data_iterator['label']
    car_model_by_label = label_dic[str(label)]
    plt.title(f'Image Label: {car_model_by_label} ({label})')
    plt.imshow(image)


random_car = cars_train.as_numpy_iterator().next()

plot_single_example(random_car)


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
    label = tf.one_hot(example['label'], 196)

    return image, label


data_augmentation = tf.keras.Sequential(
    [layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomBrightness(0.05),
        layers.RandomContrast(0.5, 1.5), layers.RandomZoom(0.2, 0.2), layers.RandomTranslation(0.2, 0.2),
        layers.Lambda(lambda x: tfa.image.gaussian_filter2d(x, (3, 3), 1.0)), ])

batch_size = 32
cars_train_pp = cars_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
# cars_train_pp = cars_train_pp.map(lambda x, y: (data_augmentation(x, training=True), y),
#                                   num_parallel_calls=tf.data.AUTOTUNE)
cars_val_pp = cars_val.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
cars_test_pp = cars_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

image_batch, label_batch = next(iter(cars_train_pp.take(1)))
random_index = tf.random.uniform(shape=[], minval=0, maxval=image_batch.shape[0], dtype=tf.int32)
random_image = image_batch[random_index]
random_label = label_batch[random_index]
class_index = tf.argmax(random_label)
class_index = tf.cast(class_index, tf.int32).numpy()
car_model_by_label = label_dic[str(class_index)]
plt.title(f'Image Label: {car_model_by_label} ({class_index})')
plt.imshow(random_image)

########################
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalMaxPooling2D()(x)  # GlobalAveragePooling2D
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
prediction_layer = tf.keras.layers.Dense(196, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=prediction_layer)
model.summary()

epochs = 3
learning_rate = 0.001
n_workers = 8
opt = Adam(lr=learning_rate)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)


model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
hist = model.fit(cars_train_pp, epochs=epochs, validation_data=cars_val_pp, workers=n_workers)
model.save('model')

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('New_Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('New_Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()


y_pred = model.predict(cars_test_pp)

y_true = np.concatenate([y.numpy() for x, y in cars_test_pp], axis=0)
y_true = np.argmax(y_true, axis=1)

acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
print("Test accuracy is:", acc)

num_classes = 196
target_names = [label_dic[str(i)] for i in range(num_classes)]
print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=target_names))

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = tf.keras.layers.GlobalMaxPooling2D()(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
x = tf.keras.layers.Dense(2048, activation='relu')(x)
prediction_layer = tf.keras.layers.Dense(196, activation='softmax')(x)

model_2 = Model(inputs=base_model.input, outputs=prediction_layer)
model_2.summary()

model_2.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
hist = model_2.fit(cars_train_pp, epochs=epochs, validation_data=cars_val_pp, workers=n_workers)
model_2.save('model_2')

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('New_Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('New_Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()

y_pred = model.predict(cars_test_pp)

y_true = np.concatenate([y.numpy() for x, y in cars_test_pp], axis=0)
y_true = np.argmax(y_true, axis=1)

acc = accuracy_score(y_true, np.argmax(y_pred, axis=1))
print("Test accuracy is:", acc)

num_classes = 196
target_names = [label_dic[str(i)] for i in range(num_classes)]
print(classification_report(y_true, np.argmax(y_pred, axis=1), target_names=target_names))
