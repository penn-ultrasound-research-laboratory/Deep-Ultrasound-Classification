from keras.applications.inception_v3 import InceptionV3
from keras.layers import Activation, Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.python.lib.io import file_io


def get_model(config):
    model = InceptionV3(
        include_top=False,
        weights=None,
        input_shape=config.input_shape)

    # Fetch weights from Google Cloud Storage to save download time 
    model_weights = file_io.FileIO('gs://research-storage/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', mode='rb')

    temp_model_weights = './temp_weights.h5'
    temp_weights_file = open(temp_model_weights, 'wb')
    temp_weights_file.write(model_weights.read())
    temp_weights_file.close()
    model_weights.close()

    model.load_weights(temp_model_weights)

    for layer in model.layers:
        layer.trainable = False 

    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs=model.input, outputs=x)