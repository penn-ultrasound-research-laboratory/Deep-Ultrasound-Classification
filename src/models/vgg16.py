from keras.applications.vgg16 import VGG16
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model


def get_model(config):
    model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=config.input_shape)

    # Connect output pooling to dense layer with sigmoid activation
    # Assumes binary crossentropy loss function
    # model = Model(
    #     inputs=model.inputs, outputs=Dense(
    #         2,
    #         input_shape=(512,),
    #         activation="sigmoid",
    #         name="predictions")(model.layers[-1].output)
    # )

    x = model.get_layer('block5_conv3').output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=x)
