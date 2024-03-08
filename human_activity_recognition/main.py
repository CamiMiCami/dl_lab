'''
This script was modified and further developed by Zhengyu Bao as main script both for training and hyperparameter tuning
'''
import gin
import logging

from train import Trainer
from input_pipeline import datasets
from utils import utils_params, utils_misc
import models
import evaluation.eval as eva

import wandb
import datetime

gin.enter_interactive_mode()


# setup wandb
@gin.configurable
def wandb_setup(login_key, Project_Name, Sweep_Name):
    wandb.login(key=login_key)
    return Project_Name, Sweep_Name



# gin-config
gin.parse_config_files_and_bindings(["configs/config.gin"], [])


Project_Name , Sweep_Name = wandb_setup()

# setup pipeline
ds_train_r, ds_val_r, ds_test_r = datasets.load(save_to_txt=False) #for running in GPU
#ds_train_r, ds_val_r, ds_test_r = datasets.load_dataset()  #for local run with existing txt
print("txt read")

'''for each run in a tuning task'''
def main():    
    #define run name with datetime in wandb
    Run_Name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    wandb.init(name= Run_Name)
    
    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    
    #prepare dataset, batch size, repeating times for oversampling and strength for added noise can be read from wandb configuration here
    ds_train, ds_val, ds_test = datasets.prepare(ds_train_r, ds_test_r, ds_val_r, 
                                                 batch_size=wandb.config.batch_size, 
                                                 repeats=wandb.config.input_repeats, 
                                                 std=wandb.config.input_std)
    print("datasets prepared")

    # set loggers
    utils_misc.set_loggers(run_paths["path_logs_train"], logging.INFO)
    utils_params.save_config(run_paths["path_gin"], gin.config_str())
    
    #define a model
    model_archi = models.model_archi(rec_unit=wandb.config.rec_units, 
                                     dense_units=wandb.config.dense_units, 
                                     dropout_rate=wandb.config.dropout_rate)    
    model = models.Bi_GRU(model_archi)
    model.summary()
    
    #start training route
    trainer = Trainer(model, ds_train, ds_val, run_paths, class_weights=wandb.config.class_weights)
    for _ in trainer.train():
        continue
    model.summary()
    
    #evaluate for the saved checkpoint, print test acc
    num_ckpts = trainer.total_steps/trainer.ckpt_interval
    best_checkpoint = eva.ckpts_eval(model, run_paths["path_ckpts_train"], num_ckpts, ds_test) 

#set up configuration for a single training or hyper parameter tuning
#our best value after tuning was suggested as fowllowing
sweep_config = {
    "method": "grid",
    "name": Sweep_Name,  
    "metric": {"train_acc": "train_loss", "val_acc": "val_loss"},
    #"early_terminate": {"type": "hyperband", "max_iter": 24},
    "parameters": {
        "rec_units": {"values": [64]},
        "dense_units": {"values": [32]},
        "dropout_rate": {"values": [0.25]},
        "batch_size":{"values": [300]},
        "input_std":{"values": [0.2]},                #adding noise
        "input_repeats":{"values": [2]},              #oversampling
        "class_weights": {"values": [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],          #weighted loss
                                     #[0.2, 0.2, 0.2, 0.6, 0.6, 0.5, 8.0, 8.0, 10.0, 8.0, 9.0, 9.0],
                                     ]},
        

    },
}
#pass configuration to wandb and start running
sweep_id = wandb.sweep(sweep = sweep_config, project = Project_Name)

print(sweep_id)
wandb.agent(sweep_id=sweep_id, function=main )
wandb.finish()
