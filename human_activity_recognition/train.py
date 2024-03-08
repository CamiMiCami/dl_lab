'''
This script was modified and further developed by Zhengyu Bao from train.py for 1. project to define a training procedure as a class.
'''
import gin
import tensorflow as tf
import logging
import wandb

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths, total_steps, log_interval, ckpt_interval, class_weights):
        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        

        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
        self.val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')


        self.conf_mat= None
        self.class_accuracy= None



        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.class_weights = class_weights
        

        # Summary Writer. The dir need to be changed
        self.train_summary_writer = tf.summary.create_file_writer(self.run_paths["path_ckpts_train"])
        self.val_summary_writer = tf.summary.create_file_writer(self.run_paths["path_ckpts_train"])

        # Checkpoint Manager
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.model)
        self.manager = tf.train.CheckpointManager(self.checkpoint, run_paths["path_ckpts_train"], max_to_keep=None)

    @tf.function
    ### in a training step
    def train_step(self, inputs, labels):
        with tf.GradientTape() as tape:
            # forward to get prediction
            predictions = self.model(inputs, training=True)
            flattened_labels = tf.reshape(labels, (-1,))
            flattened_predictions = tf.reshape(predictions, (-1, 12))
            
            mask = tf.not_equal(flattened_labels, 0)
            flattened_labels = tf.boolean_mask(flattened_labels, mask)
            flattened_predictions = tf.boolean_mask(flattened_predictions, mask)
            flattened_labels = flattened_labels - 1
            
            # backward to calculate loss and gradients
	    ### add weighted loss
            sample_weight = tf.where(tf.equal(flattened_labels, 0),  tf.cast(self.class_weights[0], dtype=tf.float32), tf.constant(1, dtype=tf.float32))
            for i in range(1,12):
                sample_weight = tf.where(tf.equal(flattened_labels, i), tf.cast(self.class_weights[i], dtype=tf.float32), sample_weight)
            
            loss = self.loss_object(flattened_labels, flattened_predictions, sample_weight=sample_weight) 
        gradients = tape.gradient(loss, self.model.trainable_variables)
        #tf.print("gradients:",gradients)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        
	#calculate overall training acc and class training acc
        self.train_accuracy(flattened_labels, flattened_predictions)
       
        flattened_predictions_lb = tf.argmax(flattened_predictions, axis=1)
        self.conf_mat = tf.math.confusion_matrix(flattened_labels, flattened_predictions_lb)
        self.class_accuracy = tf.linalg.diag_part(self.conf_mat) / tf.reduce_sum(self.conf_mat, axis=1)
        self.class_accuracy = tf.where(tf.math.is_finite(self.class_accuracy), self.class_accuracy, 0.0)
        #tf.print("labels:",flattened_labels)
        #tf.print("predictions:",flattened_predictions_lb)
        #tf.print("sample_weight:",sample_weight)
        return self.class_accuracy

    @tf.function
    # in a validation step
    def val_step(self, inputs, labels):
        # forward to get predictions, loss and acc
        predictions = self.model(inputs, training=False)
        flattened_labels = tf.reshape(labels, (-1,))
        flattened_predictions = tf.reshape(predictions, (-1, 12))
        mask = tf.not_equal(flattened_labels, 0)
        flattened_labels = tf.boolean_mask(flattened_labels, mask)
        flattened_predictions = tf.boolean_mask(flattened_predictions, mask)
        flattened_labels = flattened_labels - 1
        
	### add weighted loss
        sample_weight = tf.where(tf.equal(flattened_labels, 0), tf.cast(self.class_weights[0], dtype=tf.float32), tf.constant(1, dtype=tf.float32))
        for i in range(1,12):
            sample_weight = tf.where(tf.equal(flattened_labels, i), tf.cast(self.class_weights[i], dtype=tf.float32), sample_weight)
            
        t_loss = self.loss_object(flattened_labels, flattened_predictions, sample_weight=sample_weight)
        
        
        self.val_loss(t_loss)
        self.val_accuracy(flattened_labels, flattened_predictions)
        
        #flattened_predictions_lb = tf.argmax(flattened_predictions, axis=1)
        #tf.print("labels:",tf.transpose(flattened_labels))
        #tf.print("val_predictions:",tf.transpose(flattened_predictions_lb))
        return 

    #define a training procedure
    def train(self):
        for idx, (inputs, labels) in enumerate(self.ds_train):
            step = idx + 1
            class_acc=self.train_step(inputs, labels)
            #print('step:',step)

            if step % self.log_interval == 0:

                # Reset val metrics every log_interval
                self.val_loss.reset_states()
                self.val_accuracy.reset_states()
                
                print('log at step',step)
                tf.print(class_acc)

                for val_inputs, val_labels in self.ds_val:
                    self.val_step(val_inputs, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'
                logging.info(template.format(step,
                                             self.train_loss.result(),
                                             self.train_accuracy.result() * 100,
                                             self.val_loss.result(),
                                             self.val_accuracy.result() * 100))

                # wandb logging
                wandb.log({'train_acc': self.train_accuracy.result() * 100, 'train_loss': self.train_loss.result(),
                           'val_acc': self.val_accuracy.result() * 100, 'val_loss': self.val_loss.result(),
                           'step': step})


                for class_idx in range(tf.shape(class_acc)[0]):
                    class_acc_train ='class '+str(class_idx+1)
                    
                    wandb.log({class_acc_train: class_acc[class_idx]})
                    


                # Reset train metrics every log_interval
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                yield self.val_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                self.manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # Save final checkpoint
                self.checkpoint.save(file_prefix=self.run_paths["path_ckpts_train"]+"/final")

                return self.val_accuracy.result().numpy()
		
		
		
