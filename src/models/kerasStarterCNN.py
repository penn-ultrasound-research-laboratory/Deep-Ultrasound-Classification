from __future__ import print_function
import argparse
import numpy as np
import os

from constants.ultrasound import IMAGE_TYPE
from constants.model import TRAIN_TEST_VALIDATION_SPLIT
from math import floor
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator




def prepare_training_data():
    """

    Iterator will be declared with array of patients. On each call to iterator 

    An Iterator yielding tuples of (x, y) where x is a numpy array of image data 
    (in the case of a single image input) or a list of numpy arrays (in the case with additional inputs) 
    and y is a numpy array of corresponding labels. If 'sample_weight' is not None, the yielded tuples are of the form (x, y, sample_weight). If y is None, only the numpy array x is returned.
    
    Arguments:

    Returns:
        Iterator that wraps a Keras ImageGenerator

    Raises 
    """
    
    # Each patient is a batch for a specific class of image 


    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)


    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)

    pass

def preprocess_train_evaluate(path_to_manifest, image_type):

    batch_size = 128
    num_classes = 2
    epochs = 5

    # input image dimensions
    img_x, img_y = 28, 28

    # load the MNIST data set, which already splits into train and test sets for us
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape the data into a 4D tensor - (sample_number, x_img_size, y_img_size, num_channels)
    # because the MNIST is greyscale, we only have a single channel - RGB colour images would have 3
    x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
    x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
    input_shape = (img_x, img_y, 1)

    # convert the data to the right type
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices - this is for use in the
    # categorical_crossentropy loss below
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adam(),
                metrics=['accuracy'])


    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.acc = []

        def on_epoch_end(self, batch, logs={}):
            self.acc.append(logs.get('acc'))

    history = AccuracyHistory()

    model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test),
            callbacks=[history])

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    # plt.plot(range(1, 11), history.acc)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.show()

if __name__ == '__main__':

    # establish arguments -- path to manifest
    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_manifest',
        help='absolute path to source data manifest')

    parser.add_argument('image_type', 
        help='determines preprocessing channel to run',
        type=IMAGE_TYPE,
        choices=list(IMAGE_TYPE))

    arguments = parser.parse_args()

    preprocess_train_evaluate(
        arguments['path_to_manifest'],
        arguments['image_type'])



## WORK to be done according to the internet is to write a preprocessing pipeline. Each image is a batch --> randomly sample ~16 croppings per image 
