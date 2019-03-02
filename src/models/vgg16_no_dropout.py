from keras.applications.vgg16 import VGG16
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model

def get_model(config):
    model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=config.input_shape)

    for layer in model.layers:
        layer.trainable = False 

    x = model.get_layer('block5_conv3').output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=x)
