'''
This script was developed by Zhengyu Bao to calculate output for self-recorded data with a trained model.
'''
from models import Bi_GRU
from MotionSensor import sensor_record as rec
import tensorflow as tf
import numpy as np

'''call the function to record data for 15 seconds. There is also a existing recording as txt file which can be read'''
#record = rec.record_data(rec_time=15)
record = np.loadtxt("MotionSensor/record_data.txt")

'''prepare the recorded data as input'''
input_data = rec.load_record(record)


print("input prepared")

''' model & checkpoint'''

#define a model
input_shape = (250,6)
rec_unit = 64
dense_units = 32
dropout_rate = 0.25
num_classes = 12
model_archi = input_shape, rec_unit, dense_units, dropout_rate, num_classes 
model = Bi_GRU(model_archi)

#load checkpoint to the model
checkpoint = "ckpt-469"
tf.train.Checkpoint(model=model).restore(checkpoint)
model.summary()

#feed forward to calculate the output
flat_y_pred = []
for window in input_data: 
    y_pred = model.predict(window)
    y_pred = tf.reshape(y_pred, (-1, 12))
    y_pred = tf.argmax(y_pred, axis=1)
    flat_y_pred.append(y_pred)
    
print(flat_y_pred)

#plot result can be done additionally

