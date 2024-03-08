'''
This script was developed by Zhegnyu Bao to define a model.

The first function can be called by any other script to configure the model architecture very flexible.

Then there are 6 functions which can generate 6 different kinds of models
'''
import tensorflow as tf

#input_shape = (timesteps,channels)

def model_archi(rec_unit, dense_units, dropout_rate, input_shape = (250,6), num_classes=12):
    model_archi = input_shape, rec_unit, dense_units, dropout_rate, num_classes 
    return model_archi


def vanilla_RNN(model_archi):

    input_shape, rec_unit, dense_units, dropout_rate, num_classes = model_archi
    
    inputs = tf.keras.Input(shape = input_shape)
    
    # Recurrent layer + 2 dense layers
    out = tf.keras.layers.SimpleRNN(units=rec_unit, activation='relu', return_sequences=True)(inputs)
    out = tf.keras.layers.Dense(dense_units, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(out)
    
    return tf.keras.Model(inputs = inputs, outputs = outputs, name = 'vanilla_RNN')



def BRNN(model_archi):

    input_shape, rec_unit, dense_units, dropout_rate, num_classes = model_archi
    
    inputs = tf.keras.Input(shape = input_shape)
    
    # Recurrent layer + 2 dense layers
    out = tf.keras.layers.Bidirectional(tf.keras.layers.SimpleRNN(units=rec_unit, activation='relu', return_sequences=True))(inputs)
    out = tf.keras.layers.Dense(dense_units, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(out)
    
    return tf.keras.Model(inputs = inputs, outputs = outputs, name = 'BRNN')




def LSTM(model_archi):
    input_shape, rec_unit, dense_units, dropout_rate, num_classes = model_archi
    inputs = tf.keras.Input(shape=input_shape)
    
    # LSTM layer + 2 dense layers
    out = tf.keras.layers.LSTM(units=rec_unit, activation='relu', return_sequences=True)(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dense(units=dense_units, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(out)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='LSTM')



def bi_LSTM(model_archi):
    input_shape, rec_unit, dense_units, dropout_rate, num_classes = model_archi
    inputs = tf.keras.Input(shape=input_shape)
    
    # Bidirectional LSTM layer + 2 dense layers
    out = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=rec_unit, activation='relu', return_sequences=True))(inputs)
    out = tf.keras.layers.BatchNormalization()(out)
    out = tf.keras.layers.Dense(units=dense_units, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(out)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='bidirectional_LSTM')



def GRU(model_archi):
    input_shape, rec_unit, dense_units, dropout_rate, num_classes = model_archi
    inputs = tf.keras.Input(shape=input_shape)
    
    # GRU layer + 2 dense layers
    out = tf.keras.layers.GRU(units=rec_unit, activation='relu', return_sequences=True)(inputs)
    out = tf.keras.layers.Dense(units=dense_units, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(out)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='GRU')

def Bi_GRU(model_archi):
    input_shape, rec_unit, dense_units, dropout_rate, num_classes = model_archi
    inputs = tf.keras.Input(shape=input_shape)
    
    # bidirectional GRU layer + 2 dense layers
    out = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=rec_unit, activation='relu', return_sequences=True))(inputs)
    out = tf.keras.layers.Dense(units=dense_units, activation='relu')(out)
    out = tf.keras.layers.Dropout(dropout_rate)(out)
    outputs = tf.keras.layers.Dense(units=num_classes, activation='softmax')(out)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='Bi-GRU')

