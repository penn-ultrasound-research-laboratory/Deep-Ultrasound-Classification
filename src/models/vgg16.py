from keras.applications.vgg16 import VGG16

def get_model(config):
    return VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=config.input_shape,
        pooling=config.pooling)

