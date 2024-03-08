import tensorflow as tf

from layers import vgg_block


def vgg_like(base_filters, n_blocks, dense_units, dropout_rate, input_shape=(255,255,3), n_classes=2):
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

    inputs = tf.keras.Input(input_shape)
    out = vgg_block(inputs, base_filters)
    for i in range(1, n_blocks):
        out = vgg_block(out, base_filters * 2 ** (i))
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    # out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu)(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(1, activation= 'sigmoid')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="vgg_like")