'''
This script was modified from original given script and further developed by Shuaike Liu to build a vgg-like block.

The function vgg_block() is called in models.architecture.py to build a vgg block in vgg-like model. 
It reveives parameters including input of VGG block, number of filters and kernel_size.
'''
import tensorflow as tf

def vgg_block(inputs, filters, l2_strength, kernel_size= (3, 3)):
    """A single VGG block consisting of two convolutional layers, followed by a max-pooling layer.
    Parameters:
        inputs (Tensor): input of the VGG block
        filters (int): number of filters used for the convolutional layers
        kernel_size (tuple: 2): kernel size used for the convolutional layers, e.g. (3, 3)
    Returns:
        (Tensor): output of the VGG block
    """

    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(inputs)
    out = tf.keras.layers.Conv2D(filters, kernel_size, padding='same', activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(out)
    out = tf.keras.layers.MaxPool2D((2, 2))(out)

    return out
