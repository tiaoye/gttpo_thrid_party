import pandas as pd
import yaml
import argparse
import torch
from GTPPO import GTPPO
from my_logger import Logger
import logging
from torch.utils.tensorboard import SummaryWriter

# hyperparameters
CONFIG_FILE_PATH = 'config/GTPPO_ethucy.yaml' # yaml config file containing all the hyperparameters 
EXPERIMENT_NAME = 'GTPPO_eth_0410_intention' # arbitrary name for this experiment 
DATASET_NAME = 'eth' # one of eth, hotel, univ, zara1, zara2
TRAIN_DATA_PATH = 'data/ethucy_sgan/' + DATASET_NAME + '/train/'
VAL_DATA_PATH = 'data/ethucy_sgan/' + DATASET_NAME + '/val/'
OUTPUT_DIR = 'output/' + EXPERIMENT_NAME + '/'

# load parameters from yaml file
with open(CONFIG_FILE_PATH, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# init logger
logger = Logger(OUTPUT_DIR + 'log.txt')
writer = SummaryWriter(OUTPUT_DIR + 'tensorboard/')

# init model
model = GTPPO(params)

# start training
model.train(TRAIN_DATA_PATH, VAL_DATA_PATH, params, EXPERIMENT_NAME, logger, writer, OUTPUT_DIR,dataset_name=DATASET_NAME)