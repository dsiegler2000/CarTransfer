from keras_applications.imagenet_utils import _obtain_input_shape
from keras import backend as K
from keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate, Dropout
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.utils import get_file

# Define some commonly used names
sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

WEIGHTS_PATH = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/" \
               "squeezenet_weights_tf_dim_ordering_tf_kernels.h5"
WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/" \
                      "squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"


def fire_module(x, fire_id, squeeze=16, expand=64):
    """
    Creates one fire module
    :param x: Input
    :param fire_id: ID
    :param squeeze: First convolution depth
    :param expand: Second convolution depth
    :return: The full fire module
    """

    # Make the ID
    s_id = f"fire{str(fire_id)}/"

    # Quick check
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    # Squeeze convolution
    x = Convolution2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    # Left convolution
    left = Convolution2D(expand, (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    # Right convolution
    right = Convolution2D(expand, (3, 3), padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    # Combine the right and the left
    x = concatenate([left, right], axis=channel_axis, name=s_id + 'concat')

    return x


def SqueezeNet(include_top=True, weights='imagenet', input_width=227, pooling=None, classes=1000):
    """
    SqueezeNet, as defined here: https://arxiv.org/pdf/1602.07360.pdf
    :param include_top: Include the last classification layers if true
    :param weights: Either imagenet, or None. The weights to load in
    :param input_width: Width (and height) of the input
    :param pooling: If `include_top` is set to false, this can be either 'avg' or 'max'
    :param classes: If `include_top` is True, it sets the number of classes
    :return:
    """

    # Do some quick error checking
    if weights not in ['imagenet', None]:
        raise ValueError("The `weights` argument should be either `None` (random initialization) "
                         "or `imagenet` (pre-training on ImageNet).")

    # Quick and dirty use of this method
    input_shape = _obtain_input_shape(None, default_size=input_width, min_size=48,
                                      data_format=K.image_data_format(), require_flatten=include_top)

    # Input layer, using the Tensor workflow
    img_input = Input(shape=input_shape)

    # First convolution
    x = Convolution2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(img_input)
    x = Activation('relu', name='relu_conv1')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)

    # Fire modules
    x = fire_module(x, fire_id=2, squeeze=16, expand=64)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192)
    x = fire_module(x, fire_id=8, squeeze=64, expand=256)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256)

    if include_top or weights == 'imagenet':
        # If include_top, add the last layers
        x = Dropout(0.5, name='drop9')(x)

        # Do this name thing for loading purposes
        x = Convolution2D(classes, (1, 1), padding='valid', name='conv10' + f'_{classes}' if classes != 1000 else "")(x)
        x = Activation('relu', name='relu_conv10' + f'_{classes}' if classes != 1000 else "")(x)
        x = GlobalAveragePooling2D(name='pool6' + f'_{classes}' if classes != 1000 else "")(x)
        x = Activation('softmax', name='loss' + f'_{classes}' if classes != 1000 else "")(x)
    else:
        # Otherwise, just add the pooling layer
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)
        elif pooling == None:
            pass
        else:
            raise ValueError("Unknown argument for `pooling`=" + pooling)

    # Make the actual model
    model = Model(img_input, x, name='squeezenet')

    # Load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH, cache_subdir='models')
        else:
            weights_path = get_file('squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP, cache_subdir='models')
        model.load_weights(weights_path, by_name=True)

    return model
