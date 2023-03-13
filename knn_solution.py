import keras
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.layers import Lambda
import tqdm
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
import numpy as np

cars = tfds.load('Cars196', as_supervised=False,
                 data_dir='/home/anastasia/Documents/EX/cars196/C:/Users/anast/PycharmProjects/cars196/data',
                 shuffle_files=True)
cars_dataset = cars['train'].concatenate(cars['test'])

# Split the dataset into 20/80 train-test split
train_size = int(0.8 * len(cars_dataset))
test_size = len(cars_dataset) - train_size
cars_train = cars_dataset.take(train_size)
cars_test = cars_dataset.skip(train_size)


def preprocess_image(example):
    image = example['image']
    bbox = example['bbox']
    label = example['label']

    # Crop
    h, w = tf.unstack(tf.shape(image)[:2])
    scaled_box = bbox * [h, w, h, w]
    ymin, xmin, ymax, xmax = tf.unstack(tf.cast(scaled_box, tf.int32))
    box_width = xmax - xmin
    box_height = ymax - ymin
    image = tf.image.crop_to_bounding_box(image, ymin, xmin, box_height, box_width)

    # cast
    image = tf.cast(image, tf.float32)

    # resize
    image = tf.image.resize_with_pad(image, 224, 224)

    return image, label


cars_train_pp = cars_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(
    buffer_size=tf.data.AUTOTUNE)
cars_test_pp = cars_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(
    buffer_size=tf.data.AUTOTUNE)

# KNN
model_path = '/home/anastasia/Documents/EX/cars196/best_model_tl.h5'
pretrained_model = keras.models.load_model(model_path, custom_objects={"preprocess_input": preprocess_input})
pretrained_model.summary()
feature_extractor = keras.Model(
    inputs=pretrained_model.inputs,
    outputs=pretrained_model.get_layer(name="dense_7").output,
)


def get_feature_vectors(data_set):
    x = []
    y = []
    for img, label in tqdm.tqdm(data_set):
        feature_vector = feature_extractor(img)
        x.append(tf.squeeze(feature_vector))
        y.append(tf.squeeze(label))

    x = np.array(x)
    y = np.array(y)
    return x, y


X_train, y_train = get_feature_vectors(cars_train_pp.take(100))
X_test, y_test = get_feature_vectors(cars_test_pp.take(100))
# Convert list to numm

nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)

print(nca_pipe.score(X_test, y_test))
