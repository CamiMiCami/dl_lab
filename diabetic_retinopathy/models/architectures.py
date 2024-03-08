'''
This script was modified from original given script and further developed by Shuaike Liu to build a vgg-like model.

The function vgg_like()  can be called in main.py or any other script to build a vgg-like model. 
It reveives parameters including number of base_filters, number of blocks, number of dense units, dropout rate. 
Parameter n_classes is never used, since there is only one dense unit for the output layer for binary classification.
'''
import tensorflow as tf

from models.layers import vgg_block



def vgg_like(base_filters, n_blocks, dense_units, dropout_rate, l2_strength=0.01, input_shape=(255,255,3), n_classes=2):
    """Defines a VGG-like architecture.
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes, corresponding to the number of output neurons
        base_filters (int): number of base filters, which are doubled for every VGG block
        n_blocks (int): number of VGG blocks
        dense_units (int): number of dense units
        dropout_rate (float): dropout rate
    Returns:
        (keras.Model): keras model object
    """

    assert n_blocks > 0, "Number of blocks has to be at least 1."
    
    #build VGG blocks
    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters,l2_strength = l2_strength)
    for i in range(1, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i), l2_strength = l2_strength)
        
    # add global average pooling layer and dense layer
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    # out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(1, activation= 'sigmoid')(out)
    

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg_like")
