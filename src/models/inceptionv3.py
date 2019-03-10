from keras.applications.inception_v3 import InceptionV3
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model


def get_model(config):
    model = InceptionV3(
        include_top=False,
        weights='imagenet',
        input_shape=config.input_shape)

    for layer in model.layers:
        layer.trainable = False 

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=x)

    #     inception_model = InceptionV3(weights='imagenet', include_top=False)
    # x = inception_model.output
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.2)(x)

    # predictions = Dense(10, activation='softmax')(x)

    # model = Model(inputs=inception_model.input, outputs=predictions)
    # for layer in inception_model.layers:
    #     layer.trainable = False

    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.fit(x_train, y_train)

    # for i, layer in enumerate(model.layers):

    #     if i < 249:
    #         layer.trainable = False
    #     else:
    #         layer.trainable = True
