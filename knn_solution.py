import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import numpy as np
from keras.applications.vgg16 import preprocess_input
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier

# Load cars196 dataset
cars = tfds.load('Cars196', as_supervised=False,
                 data_dir='/home/anastasia/Documents/EX/cars196/C:/Users/anast/PycharmProjects/cars196/data',
                 shuffle_files=True)
cars_dataset = cars['train'].concatenate(cars['test'])

# Split the dataset into 20/80 train-test split
train_size = int(0.8 * len(cars_dataset))
test_size = len(cars_dataset) - train_size
cars_train = cars_dataset.take(train_size)
cars_test = cars_dataset.skip(train_size)


# Function that crops, resizes, and formats an image
def preprocess_image(example):
    image = example['image']
    bbox = example['bbox']
    label = example['label']

    h, w = tf.unstack(tf.shape(image)[:2])
    scaled_box = bbox * [h, w, h, w]
    y_min, x_min, y_max, x_max = tf.unstack(tf.cast(scaled_box, tf.int32))
    box_width = x_max - x_min
    box_height = y_max - y_min
    image = tf.image.crop_to_bounding_box(image, y_min, x_min, box_height, box_width)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize_with_pad(image, 224, 224)
    return image, label


# Reformat the images in train and test
print('Processing datasets')
cars_train_pp = cars_train.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(
    buffer_size=tf.data.AUTOTUNE)
cars_test_pp = cars_test.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE).batch(32).prefetch(
    buffer_size=tf.data.AUTOTUNE)

# Load a pre-trained model to use as a feature extractor
print('Loading pre-trained model')
model_path = '/home/anastasia/Documents/EX/cars196/best_model_tl.h5'
pretrained_model = keras.models.load_model(model_path, custom_objects={"preprocess_input": preprocess_input})
pretrained_model.summary()
feature_extractor = keras.Model(
    inputs=pretrained_model.inputs,
    outputs=pretrained_model.get_layer(name="dense_7").output,
)


# Function that gets a feature vector representation per image in a dataset
def get_feature_vectors(data_set, feature_length):
    x = np.empty((0, int(feature_length / 2)))
    y = np.empty(0)
    for img, label in tqdm.tqdm(data_set):
        feature_vector = feature_extractor(img)
        feature_vector = feature_vector[:, ::2]  # reduce size (optional)
        x = np.vstack((x, np.array(feature_vector)))
        y = np.hstack((y, np.array(label)))
    return x, y


# Get feature representation for each image in train and test
print('Extracting feature vectors')
X_train, y_train = get_feature_vectors(cars_train_pp, feature_length=2048)
X_test, y_test = get_feature_vectors(cars_test_pp, feature_length=2048)

# Create and fit KNN model
print('Fitting KNN model')
nca = NeighborhoodComponentsAnalysis(random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])
nca_pipe.fit(X_train, y_train)
print('Model complete')

# Predict on test set
accuracy = nca_pipe.score(X_test, y_test)
print(f'Accuracy: {accuracy}')
