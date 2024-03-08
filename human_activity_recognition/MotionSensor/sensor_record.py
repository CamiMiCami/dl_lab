'''
This script was developed by Zhengyu Bao to record data with Nintendo gamepad which includes two functions.
This script doesn't make connection to the hardware itself, but read dataflow in a open sourse application called MotionSensor in windows. 
This application was downloaded online to connect the Hw.

After opening the MotionSensor.exe, function record_data(rec_time) can read the dataflow for certain time which is defined by parameter rec_time.

Function  load_record() loads the recorded data, makes tf dataset, performs windowing and preparing for recorded data as input
'''
import time
import pywinauto
import numpy as np
from scipy.interpolate import interp1d
import tensorflow as tf

###from sensor to data 30s data
def record_data(rec_time = 10):
    print("recording start")
    pywinauto.sysinfo.threadmode = 0
    pywinauto.sysinfo.is_x64_OS = lambda: True
    
    '''find the window to read dataflow automatically''' 
    app = pywinauto.Application().connect(title='Gyro Data Debug')
    window = app.top_window()
    
    timestamps = np.empty((0,1) ,float)
    data = np.empty((0, 6), float)
    
    '''read data for first time stamp'''
    gyro_z_label = window.child_window(auto_id="GyroZLabel").window_text()
    gyro_y_label = window.child_window(auto_id="GyroYLabel").window_text()
    gyro_x_label = window.child_window(auto_id="GyroXLabel").window_text()
    acc_z_label = window.child_window(auto_id="AccZLabel").window_text()
    acc_y_label = window.child_window(auto_id="AccYLabel").window_text()
    acc_x_label = window.child_window(auto_id="AccXLabel").window_text()
    timestamp  = time.time()
    timestamp_final = timestamp + rec_time
    
    signal_step = [gyro_x_label, gyro_y_label, gyro_z_label, acc_x_label, acc_y_label, acc_z_label]
    
    data = np.vstack((data, signal_step))
    timestamps = np.vstack((timestamps, timestamp))
    
    '''when the recording is still ongoing, trace the dataflow for any change'''
    while timestamp <= timestamp_final:
        # 定位 Static 元素
        gyro_z_label_new = window.child_window(auto_id="GyroZLabel").window_text()
        '''if data changed in dataflow of the application, record a new data sample and timestamp'''
        if gyro_z_label != gyro_z_label_new:
            gyro_z_label = window.child_window(auto_id="GyroZLabel").window_text()
            gyro_y_label = window.child_window(auto_id="GyroYLabel").window_text()
            gyro_x_label = window.child_window(auto_id="GyroXLabel").window_text()
            acc_z_label = window.child_window(auto_id="AccZLabel").window_text()
            acc_y_label = window.child_window(auto_id="AccYLabel").window_text()
            acc_x_label = window.child_window(auto_id="AccXLabel").window_text()
    
            timestamp  = time.time()
            
            signal_step = [gyro_x_label, gyro_y_label, gyro_z_label, acc_x_label, acc_y_label, acc_z_label]
            data = np.vstack((data, signal_step))
            timestamps = np.vstack((timestamps, timestamp))
    print("recording finish")
    
    '''After recording, do the regression and resampling with 50Hz, then return'''
    interp_func = interp1d(timestamps.flatten(), data.T, kind='cubic', fill_value='extrapolate')

    new_timestamps = np.arange(timestamps[0,0], timestamps[-1,0], 1/50)

    interpolated_data = interp_func(new_timestamps).T
    
    '''if needed, recorded data can be saved as txt'''
    #txt = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    #np.savetxt(txt, interpolated_data, fmt='%f', delimiter='\t')
    
    return interpolated_data
    

def load_record(sensor_data):
    '''load the recorded data, make tf dataset, windowing and prepare as input'''
    dataset = tf.data.Dataset.from_tensor_slices(sensor_data)
    dataset = dataset.window(size=250, shift=250, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(250))
    dataset = dataset.batch(300)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    
    return dataset
    
