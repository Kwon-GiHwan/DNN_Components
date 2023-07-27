#!pip install wandb -qU
import wandb
import os
class WanDB():
    def __init__(self):
        self.config = {
            'epochs': 5,
            'classes': 10,
            'batch_size': 128,
            'kernels': [16, 32],
            'weight_decay': 0.0005,
            'learning_rate': 1e-3,
            'dataset': 'MNIST',
            'architecture': 'CNN',
            'val_evrey': 5,
            'seed': 42
        }

        self.project_name = ''
        self.entity_name = ''
        self.log_mode = 'all'
        self.log_freq = 10

        # # os.environ['WANDB_MODE'] = 'offline'
        # # This is secret and shouldn't be checked into version control
        # WANDB_API_KEY =$YOUR_API_KEY
        # # Name and notes optional
        # WANDB_NAME = "My first run"
        # WANDB_NOTES = "Smaller learning rate, more regularization."
        # WANDB_ENTITY =$username
        # WANDB_PROJECT =$project

        self.wd = wandb
        self.wd.login()

    def init(self):
        self.wd.init(project=self.project_name, entity=self.entity_name, config=self.config)

    def watch(self, model, criterion):
        self.wd.watch(model, criterion, log=self.log_mode, log_freq=self.log_freq)

    def log(self, log_msg):
        self.wd.log(log_msg)