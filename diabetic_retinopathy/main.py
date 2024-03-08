'''
This script was developed by Zhengyu Bao as main script both for training and hyperparameter tuning
'''
import gin
import logging

from train import Trainer
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import vgg_like
from models2.resnet import build_resnet
from models3.resnet50 import build_ResNet50

import evaluation as eva

import wandb
import datetime

gin.enter_interactive_mode()

# setup wandb and Sweep Configurations
@gin.configurable
def wandb_setup(login_key, Project_Name, Sweep_Name):
    wandb.login(key=login_key)
    return Project_Name, Sweep_Name

# gin-config
gin.parse_config_files_and_bindings(["configs/config.gin"], [])
    
Project_Name , Sweep_Name = wandb_setup()


'''for each run in a tuning task'''
def main():
    #define run name with datetime in wandb
    Run_Name = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")
    wandb.init(project= "test project", name= Run_Name)
    
    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    
    #load and prepare dataset, 
    #parameters for different augmentation methods can be read from wandb configuration here
    ds_train_r, ds_val_r, ds_test_r, ds_info_r = datasets.load(seed = wandb.config.seed)
    ds_train, ds_val, ds_test, ds_info= datasets.prepare(ds_train_r, ds_val_r, ds_test_r, ds_info_r, 
                                                         apply_flip=wandb.config.apply_flip,
                                                         x_bright=wandb.config.x_bright, 
                                                         x_contrast=wandb.config.x_contrast, 
                                                         x_saturation=wandb.config.x_saturation, 
                                                         x_quality=wandb.config.x_quality)
    # set loggers
    utils_misc.set_loggers(run_paths["path_logs_train"], logging.INFO)
    
    #define a model
    if wandb.config.model_typ == 'vgg_like':
        model = vgg_like(base_filters = wandb.config.vgg_base_filters,
                         n_blocks = wandb.config.vgg_n_blocks, 
                         dense_units = wandb.config.vgg_dense_units, 
                         dropout_rate = wandb.config.dropout_rate,
                         l2_strength=wandb.config.l2_str)
        
    elif wandb.config.model_typ == 'resnet':
        model = build_resnet(n_blocks = wandb.config.resnet_n_blocks,
                             base_filters = wandb.config.resnet_base_filters,
                             l2_strength=wandb.config.l2_str,
                             dropout_rate = wandb.config.dropout_rate)
        
    elif wandb.config.model_typ == 'pretrained_resnet':
        model = build_ResNet50(dense_units=wandb.config.pretrained_resnet_dense_units)
    
    else:
        print("ERROR: please choose model between vgg_like, resnet and pretrained_resnet")
        
    model.summary()
    
    #start training route
    trainer = Trainer(model, ds_train, ds_val, ds_info, run_paths, learning_rate = wandb.config.learning_rate)
    for _ in trainer.train():
        continue
    
    #evaluate for the saved checkpoint, print test acc
    num_ckpts = trainer.total_steps/trainer.ckpt_interval
    best_checkpoint = eva.ckpts_eval(model, run_paths["path_ckpts_train"], num_ckpts, ds_test)
    
#set up configuration for a single training or hyper parameter tuning
#our best value after tuning was suggested as fowllowing
sweep_config = {
    "method": "grid",
    "name": Sweep_Name,  
    "description" : "this is a test" ,
    "metric": {"train_acc": "train_loss", "val_acc": "val_loss"},
    "parameters": {
        "model_typ": {"values": ['resnet']},  #choose between 'vgg_like', 'resnet' and 'pretrained_resnet'
        "vgg_base_filters": {"values": [8]},
        "vgg_n_blocks": {"values": [4]},
        "vgg_dense_units": {"values": [32]},
        
        "resnet_n_blocks": {"values": [8]},
        "resnet_base_filters": {"values": [8]},
        
        "pretrained_resnet_dense_units": {"values": [64]},
        
        "apply_flip": {"values": [True]},
        "x_bright": {"values": [0]},
        "x_contrast": {"values": [0.2]},
        "x_saturation": {"values": [0.2]},
        "x_quality": {"values": [100]},
        
        "seed":{"values": [47]},
        
        "learning_rate":{"values": [0.001]},
        "dropout_rate": {"values": [0.3]},
        "l2_str":{"values": [0.01]}
    },
}

#pass configuration to wandb and start running
sweep_id = wandb.sweep(sweep = sweep_config, project = Project_Name)

print(sweep_id)
wandb.agent(sweep_id=sweep_id, function=main )
wandb.finish()
