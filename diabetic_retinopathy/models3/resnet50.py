'''
This script was developed by Zhengyu Bao to build a pretrained resnet. 

The function build_ResNet50() can be called by main.py or any other script to build a pretrained resnet 50. 
It receives parameters including number of dense units, dropout rate as well as strength of l2 regularization.
'''
import tensorflow as tf

def build_ResNet50(dense_units, dropout_rate=0.3, input_shape= (255,255,3), l2_strength=0):
    
    #pretrained resnet from tf.keras.application
    base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    out = base_model(inputs, training=False)  

    #add global average pooling and two dense layers
    out = tf.keras.layers.GlobalAveragePooling2D()(out)
    out = tf.keras.layers.Dense(dense_units, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(l2_strength))(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet_transfer_learning")
