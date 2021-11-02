from warnings import filters
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from keras import optimizers
from keras.preprocessing import image
from tensorflow.python.ops.gen_math_ops import imag
import numpy as np


def DataPreprocessing():
    """
    We apply transformation on the training set, with the intent to avoid overfitting.
    This is also called image augmentation
    """
    train_data_generation = ImageDataGenerator(rescale=1./255,  # divides each pixel by value of 255
                                               shear_range=0.2,
                                               zoom_range=0.2,
                                               horizontal_flip=True)
    training_set = train_data_generation.flow_from_directory(
        "/media/esteban/TOSHIBA EXT/DeepLearning/P16-Colab-Changes/DL Colab Changes/Convolutional_Neural_Networks 3/dataset/training_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary")

    """
    Preprocessing the Test Data set
    """
    test_data_generation = ImageDataGenerator(rescale=1./255)
    test_set = test_data_generation.flow_from_directory(
        "/media/esteban/TOSHIBA EXT/DeepLearning/P16-Colab-Changes/DL Colab Changes/Convolutional_Neural_Networks 3/dataset/test_set",
        target_size=(64, 64),
        batch_size=32,
        class_mode="binary")

    return training_set, test_set


def BuildingCNN():

    cnn = tf.keras.Sequential()  # initialises the CNN, sequence of layers
    # Add Layers, Filters (Number of feature detectors)
    cnn.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, activation="relu", input_shape=[64, 64, 3]))

    # Pooling
    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    """ADD SECOND LAYER"""
    cnn.add(tf.keras.layers.Conv2D(
        filters=32, kernel_size=3, activation="relu"))

    cnn.add(tf.keras.layers.MaxPooling2D(pool_size=2, strides=2))

    """ FLATTENING"""
    cnn.add(tf.keras.layers.Flatten())

    """ Connection of Dense"""
    # unit is number of neurons
    cnn.add(tf.keras.layers.Dense(units=128, activation="relu"))

    """ Output layer"""
    cnn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    return cnn


def TrainingCNN(model, data):
    cnn = model
    training_set, test_set = data

    """ COMPILE CNN"""

    # cnn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    #             loss=tf.keras.losses.BinaryCrossentropy(),
    #             metrics=[tf.keras.metrics.Accuracy()])
    cnn.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])
    """FIt into the Test Set"""
    cnn.fit(x=training_set, validation_data=test_set, epochs=25)
    return cnn


def SinglePrediction(model, data):
    test_image = image.load_img(
        "/media/esteban/TOSHIBA EXT/DeepLearning/P16-Colab-Changes/DL Colab Changes/Convolutional_Neural_Networks 3/dataset//single_prediction/cat_or_dog_1.jpg",
        target_size=(64, 64))

    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = model.predict(test_image/255.0)
    data[0].class_indices  # allows to perform the encoding of the result

    if result[0][0] > 0.5:
        prediction = "dog"
    else:
        prediction = "cat"

    return prediction


def main():
    data = DataPreprocessing()
    cnn = BuildingCNN()
    trained_cnn = TrainingCNN(cnn, data)
    print(SinglePrediction(trained_cnn, data))


if __name__ == "__main__":
    main()
