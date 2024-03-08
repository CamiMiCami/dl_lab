'''
This Script includes two functions. 

Function evaluate() was developed by Shuaike Liu. 
It is used to load a saved checkpoint to a model and evaluate it with test dataset. Test accuracies overall and for each classes are returned

Function ckpts_eval() was developed by Zhengyu Bao.
It calls evaluate() function for all the checkpoints saved in a folder and find the highest test accuracy to return.
'''
import tensorflow as tf
import numpy as np
import logging


def evaluate(model, checkpoint, ds_test):
    
    # load checkpoint into the model
    status = tf.train.Checkpoint(model=model).restore(checkpoint)
    status.expect_partial()
    
    #feed forward, calculate output for test dataset
    flat_y_true = []
    flat_y_pred = []
    
    for windows, labels in ds_test: 
        y_true = tf.reshape(labels, (-1,))
        y_pred = model.predict(windows)
        y_pred = tf.reshape(y_pred, (-1, 12))
        
        mask = tf.not_equal(y_true, 0)
        y_true = tf.boolean_mask(y_true, mask)
        y_pred = tf.boolean_mask(y_pred, mask)
        y_true = y_true - 1
        
        y_pred = tf.argmax(y_pred, axis=1)
        flat_y_pred.append(y_pred)
        flat_y_true.append(y_true)
    
    #calculate confution matrix with prediction and true label, then calculate overall accuracy and class accuracy and return
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred)

    true_positives = tf.linalg.diag_part(conf_matrix)
    class_totals = tf.reduce_sum(conf_matrix, axis=1)
    accuracy = tf.reduce_sum(true_positives[1:]) / tf.reduce_sum(class_totals[1:])
    class_accuracy = true_positives / class_totals

    return accuracy.numpy(), class_accuracy.numpy()
    
    
def ckpts_eval(model, ckpts_folder, num_ckpts, ds_test):
    ckpts_index = 1
    highest_name = ""
    highest_acc = 0
    
    #for all the checkpoints in the folder, evaluate them, log the accuracies in log file and also print out. And also compare the accuracies in the same time
    while ckpts_index <= num_ckpts:
        checkpoint_name = "/ckpt-"+str(ckpts_index)
        checkpoint = ckpts_folder + checkpoint_name
        acc, class_acc = evaluate(model, checkpoint, ds_test)
        print(f'checkpoint{checkpoint_name}, \n Overall_acc: {acc}, \n Class_acc: \n {class_acc}.')
        logging.info(f'checkpoint{checkpoint_name}, \n Overall_acc: {acc}, \n Class_acc: {class_acc}.')
        ckpts_index += 1
        if acc>highest_acc:
            highest_acc = acc
            highest_name = checkpoint_name
        
    #call evaluate() also for the final checkpoint.
    checkpoint_name = "/final-"+str(ckpts_index)
    checkpoint = ckpts_folder + checkpoint_name
    metrics = evaluate(model, checkpoint, ds_test)
    print(f'checkpoint{checkpoint_name}, \n Overall_acc: {acc}, \n Class_acc: \n {class_acc}.')
    logging.info(f'checkpoint{checkpoint_name},\n Overall_acc,\n Class_acc: {metrics}.')
    print("the highest test accuracy is",highest_acc,"for the checkpoint",highest_name)
    logging.info(f'the highest test accuracy is{highest_acc},for the checkpoint{highest_name}')
    
    return ckpts_folder+highest_name
