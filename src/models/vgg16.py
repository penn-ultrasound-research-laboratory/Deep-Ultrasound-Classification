from keras.applications.vgg16 import VGG16
from keras.layers import Dense
from keras.models import Model


def get_model(config):
    model = VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=config.input_shape,
        pooling=config.pooling)

    # Connect output pooling to dense layer with sigmoid activation
    # Assumes binary crossentropy loss function
    model = Model(
        inputs=model.inputs, outputs=Dense(
            2,
            input_shape=(512,),
            activation="sigmoid",
            name="predictions")(model.layers[-1].output)
    )

    return model
