import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D

def get_session():
    """ Limit session memory usage
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

def tfSummary(tag, val):
    """ Scalar Value Tensorflow Summary
    """
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])


def conv_layer(self, d):
    """ Returns a 2D Conv layer, with L2-regularization and ReLU activation
    """
    return Conv2D(d, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')

def conv_block(self, inp, d, pool_size=(2, 2)):
    """ Returns a 2D Conv block, with a convolutional layer, max-pooling,
    dropout and batch-normalization
    """
    conv = self.conv_layer(d)(inp)
    return MaxPooling2D(pool_size=pool_size)(conv)
