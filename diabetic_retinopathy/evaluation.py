'''
This script was developed by Zhengyu Bao which includes two functions

Function evaluate() is used to load a saved checkpoint to a model and evaluate it with test dataset. Test accuracies and confusion matrix are returned

Function ckpts_eval() calls evaluate() function for all the checkpoints saved in a folder and find the highest test accuracy to return.
'''
import tensorflow as tf
import logging


#function to load a checkpoint and return accuracy 
def evaluate(model, checkpoint, ds_test):
    #load checkpoint into the model
    status = tf.train.Checkpoint(model=model).restore(checkpoint)
    status.expect_partial()
    
    tp,tn,fp,fn = 0,0,0,0
    
    #feed forward to calculate confusion matrix
    for inputs, y_true in ds_test:
        y_pred = model(inputs, training=False)
        y_true = tf.cast(y_true, tf.bool)
        y_pred = tf.cast(tf.math.round(y_pred), tf.bool)
        

        true_positives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, y_pred), tf.int32))
        true_negatives = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true), tf.logical_not(y_pred)), tf.int32))
        false_positives = tf.reduce_sum(tf.cast(tf.logical_and(tf.logical_not(y_true), y_pred), tf.int32))
        false_negatives = tf.reduce_sum(tf.cast(tf.logical_and(y_true, tf.logical_not(y_pred)), tf.int32))
        
        
        tp+=true_positives.numpy()
        tn+=true_negatives.numpy()
        fp+=false_positives.numpy()
        fn+=false_negatives.numpy()
        
    acc = (tp + tn) / (tp + tn + fp + fn)
    
    metrics = tp, tn, fp, fn, acc

    return metrics


#function to get test_acc after training for all checkpoints and save as txt
def ckpts_eval(model, ckpts_folder, num_ckpts, ds_test):
    ckpts_index = 1
    highest_name = ""
    highest_acc = 0
    

    #for all the checkpoints in the folder, evaluate them, log the accuracies in log file and also print out. And also compare the accuracies in the same time
    while ckpts_index <= num_ckpts:
        
        checkpoint_name = "/ckpt-"+str(ckpts_index)
        print(checkpoint_name)
        checkpoint = ckpts_folder + checkpoint_name
        metrics = evaluate(model, checkpoint, ds_test)
        print(metrics)
        logging.info(f'checkpoint{checkpoint_name},tp, tn, fp, fn, acc: {metrics}.')
        
        if metrics[-1]>highest_acc:
            highest_acc = metrics[-1]
            highest_name = checkpoint_name
        ckpts_index += 1
    
    #call evaluate() also for the final checkpoint.
    checkpoint_name = "/final-"+str(ckpts_index)
    print(checkpoint_name)
    checkpoint = ckpts_folder + checkpoint_name
    metrics = evaluate(model, checkpoint, ds_test)
    print(metrics)
    logging.info(f'checkpoint{checkpoint_name},tp, tn, fp, fn, acc: {metrics}.')
    print("the highest test accuracy is",highest_acc,"for the checkpoint",highest_name)
    
    return ckpts_folder+highest_name

    
        
