# GENERAL LIBRARIES
import os
import argparse
from datetime import datetime
# MACHINE LEARNING LIBRARIES
import numpy as np
import torch
# CUSTOM LIBRARIES
from utils.tools import read_yaml, Logger
from utils.trainer import Trainer

# LOAD CONFIG 
parser = argparse.ArgumentParser(description='Process some input')
parser.add_argument('--config', default='utils/config.yaml', type=str, help='Config path', required=False)   
parser.add_argument('--benchmark','-b', action='store_true', help='Run a benchmark') 

args = parser.parse_args()
config = read_yaml(args.config)

for entry in ['MODEL_DIR','RESULTS_DIR','LOG_DIR']:
    if not os.path.exists(config[entry]):
        os.mkdir(config[entry])

now = datetime.now()
logger = Logger(config['LOG_DIR']+now.strftime("%y%m%d%H%M%S")+'.txt')

# SET DEVICE
device = torch.device(f'cuda:{config["GPU"]}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)

# SET TRAINER
trainer = Trainer(config, logger)

if args.benchmark:
    # RUN BENCHMARK
    trainer.do_benchmark()

