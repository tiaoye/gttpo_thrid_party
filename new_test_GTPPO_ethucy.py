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
TEST_DATA_PATH = 'data/ethucy_sgan/' + DATASET_NAME + '/test/'
OUTPUT_DIR = 'output/' + EXPERIMENT_NAME + '/'
MODEL_PATH = 'output\GTPPO_eth_0410_intention\GTPPO_eth_0410_intention_weights.pt'
# load parameters from yaml file
with open(CONFIG_FILE_PATH, 'r') as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# init logger
logger = Logger(OUTPUT_DIR + 'log.txt')
writer = SummaryWriter(OUTPUT_DIR + 'tensorboard/')

# init model
model = GTPPO(params)

# load model
model.load(MODEL_PATH,logger)
# start training
model.evaluate(TEST_DATA_PATH, params, logger, writer, dataset_name=DATASET_NAME)