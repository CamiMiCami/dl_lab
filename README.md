# Team15
- Shuaike Liu (st180457)
- Zhengyu Bao (st186295)

# How to run the code

## 1. diabetic_retinopathy (under folder diabetic_retinopathy)

### 1.1. training or tuning 
1.1.1. setup in config file: ~/configs/config.gin including training steps, dataset dir and wandb user info 

1.1.2. setup in script: ~/main.py in line 92:112 including all the hypermeter setting. The values in the script are suggested 

1.1.3. run main.py 

1.1.4. training and evaluation info as well as model summary is printed and logged

### 1.2. visualization
1.2.1. run grad_cam_vgg.py with your own image path for vgg grad-cam

1.2.2. run grad_cam_resnet.py with your own image pyth for resnet grad-cam

## 2. human_activity_recognition (under folder human_activity_recognition) 
### 2.1. training or tuning 
2.1.1. setup in config file: ~/configs/config.gin including training steps, dataset dir and wandb user info 

2.1.2. setup in script: ~/main.py in line 83:92 including all the hypermeter setting the values in the script now are suggested 

2.1.3. run main.py 

2.1.4. training and evaluation info as well as model summary is printed and logged

### 2.2. visualization of result
2.2.1 modify prepare_visu_data.py with your own file dir and label path.

2.2.2 modify prepare_visu_data.py to match your own acc_pattern and gyro_pattern.

2.2.3 run visualization_true.py with the .txt filenames you defined previously to show true label visualization.

2.2.4 run predict.py with the .txt filenames you defined previously to generate predict file and show accuracy.

2.2.5 run visualization_predict.py with the .txt filenames you defined previously to show predicted label visualization.

### 2.3. record own data with Nintendo gamepad 
2.3.1. with play_joycon.py, data can be recorded by a user with joycon gamepad. The recording can be used for recognition as an application of our project in the real world. A txt file including record data is also provided. 

2.3.2 run play_joycon.py prediction is printed out. And plotting can be done additionally.

# Results
1. In the first project, with augmentaion methods, as well as l2 regularization and so on, test accuracy of a vgg-like model can reach 84.5% (highest) and 82.5% (average over 3 most recent runs). While test accuracy of a resnet model can reach 87.4% (highest) and 86.4% (average over 3 most recent runs).

2. In the second project, with bidirectional GRU model, with methods oversampling and adding noise, the overall test accuracy can reach 91.8% and balaced accuracy can reach 82.8%. While with methods of weighted loss calculation and adding noise, the overall test accuracy can reach 89.2% and balaced accuracy can reach 85.2%.


