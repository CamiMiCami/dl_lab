'''
This script was developed by Shuaike Liu to build a resnet. It includes 2 functions residual_block() and build_resnet()

Function residual_block() is to build blocks for resnet. It is called in function build_resnet()

Function build_resnet() can be called by main.py or any other script to build a resnet. 
It receives parameters including number of blocks, number of base filters as well as strength of l2 regularization.
'''
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, AveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2


def residual_block(x, filters, kernel_size=3, strides=1, l2_strength=0.01):
    # Shortcut connection
    shortcut = x

    # First convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = BatchNormalization()(x)

    if strides != 1 or x.shape[-1] != shortcut.shape[-1]:
        shortcut = Conv2D(filters, 1, strides=strides, padding='valid', kernel_regularizer=l2(l2_strength))(shortcut)
        shortcut = BatchNormalization()(shortcut)

    # Element-wise addition of shortcut and residual
    x = Add()([x, shortcut])
    x = ReLU()(x)

    return x


def build_resnet(n_blocks, base_filters, dropout_rate, input_shape=(255,255,3),  l2_strength=0.01):

    inputs = Input(shape=input_shape)

    x = inputs

    # Initial convolutional layer
    x = Conv2D(base_filters, 3, padding='same', kernel_regularizer=l2(l2_strength))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    for i in range(1, n_blocks):
        if i % 2 == 0:
            x = residual_block(x, base_filters * 2 * (i), l2_strength = l2_strength)
            x = tf.keras.layers.MaxPool2D((2, 2))(x)
        else:
            x = residual_block(x, base_filters * 2 * (i), l2_strength=l2_strength)
        #x = residual_block(x, base_filters * 2 * (i), l2_strength = l2_strength)

    # Global Average pooling and final dense layer 
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    #x = AveragePooling2D(4)(x) ###If using average pooling, number of parameter would be too big
    #x = Flatten()(x)
    
    
    x = Dense(32, activation='relu', kernel_regularizer=l2(l2_strength))(x)
    x = Dropout(dropout_rate)(x)
    
    outputs = Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model

