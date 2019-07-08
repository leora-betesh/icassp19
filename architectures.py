
from keras.layers import Dense, Input, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten, Activation
from keras.models import Model
from keras.regularizers import l2
import keras
keras.backend.set_image_data_format('channels_first')

# =====================================================================================
def get_vggish_model(input_shape=None, out_dim=128, path_weights=None, params_extract=None):
    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))
    aud_input = Input(shape=input_shape, name='input_1')
    # Block 1
    x = Conv2D(64, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv1')(aud_input)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1')(x)

    # Block 2
    x = Conv2D(128, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2')(x)

    # Block 3
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_1')(x)
    x = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv3/conv3_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3')(x)

    # Block 4
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_1')(x)
    x = Conv2D(512, (3, 3), strides=(1, 1), activation='relu', padding='same', name='conv4/conv4_2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4')(x)

    # FC block
    x = Flatten(name='flatten_')(x)
    x = Dense(4096, activation='relu', name='vggish_fc1/fc1_1')(x)
    x = Dense(4096, activation='relu', name='vggish_fc1/fc1_2')(x)
    out = Dense(out_dim, activation='relu', name='vggish_fc2')(x)

    model = Model(inputs=aud_input, outputs=out, name='VGGish')
    model.load_weights(path_weights)
    for layer in model.layers[:8]:
        layer.trainable = False

    out_vggish = model.get_layer('vggish_fc2').output

    out = Dense(3, kernel_initializer='he_normal', kernel_regularizer=l2(1e-3), activation='softmax', name='prediction')(out_vggish)
    model = Model(model.input, out)
    model.summary()
    return model

def get_model_baseline(params_learn=None, params_extract=None):
    """

    :param params_learn:
    :param params_extract:
    :return:
    """

    input_shape = (1, params_extract.get('patch_len'), params_extract.get('n_mels'))

    channel_axis = 1
    n_class = params_learn.get('n_classes')

    spec_start = Input(shape=input_shape)
    spec_x = spec_start

    # l1
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(24, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)

    # l2
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)
    spec_x = MaxPooling2D(pool_size=(4, 2), data_format="channels_first")(spec_x)

    # l3
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Conv2D(48, (5, 5),
                    padding='same',  # fmap has same size as input
                    kernel_initializer='he_normal',
                    data_format='channels_first')(spec_x)
    spec_x = BatchNormalization(axis=1)(spec_x)
    spec_x = Activation('relu')(spec_x)

    spec_x = Flatten()(spec_x)
    spec_x = Dropout(0.5)(spec_x)
    spec_x = Dense(64,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-3),
                   activation='relu',
                   name='dense_1')(spec_x)

    spec_x = Dropout(0.5)(spec_x)
    out = Dense(n_class,
                kernel_initializer='he_normal',
                kernel_regularizer=l2(1e-3),
                activation='softmax',
                name='prediction')(spec_x)

    model = Model(inputs=spec_start, outputs=out, name='Baseline')

    return model
