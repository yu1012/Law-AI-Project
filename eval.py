import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import datasets
import transformers 
import numpy as np
import pytorch_lightning as pl
import pandas as pd

from tqdm import tqdm
import pickle
from sklearn.metrics import average_precision_score, roc_auc_score

import warnings
warnings.filterwarnings(action='ignore')

from transformers import AutoModel, AutoTokenizer

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers

from data import DataModule
from model import Classifier

import wandb


torch.set_num_threads(6)

backbone = AutoModel.from_pretrained('BM-K/KoSimCSE-roberta-multitask') 
tokenizer = AutoTokenizer.from_pretrained('BM-K/KoSimCSE-roberta-multitask')

with open("data.pkl", "rb") as f:
    data = pickle.load(f)

train_samples = [(sample[1], 1) for sample in data['train_pos']]
val_samples = [(sample[1], 1) for sample in data['val_pos']]
test_samples = [(sample[1], 1) for sample in data['test_pos']] + [(sample[1], 0) for sample in data['test_neg']]

pair_data = pd.read_csv('인용관계.csv')
pair_data = pair_data.dropna(axis=0)
pair_data = [[row['본발명'], row['인용발명']] for idx, row in pair_data.iterrows()]
pair_data = [(sample, 1) for sample in pair_data]

data = {'train': train_samples, 'val':val_samples, 'test':test_samples, 'pair_data': pair_data}


data_module = DataModule(tokenizer, data, 15)
model = Classifier.load_from_checkpoint(backbone=backbone, checkpoint_path="/workspace/data/AI599/my-test-project/oengq6sn/checkpoints/epoch=1-step=692.ckpt")

trainer = pl.Trainer(
    max_epochs=2,
    devices=[2],
    accelerator="gpu",
    num_sanity_val_steps=-1,
    fast_dev_run=not True, 
    limit_train_batches=1.0,
    limit_val_batches=1.0,
    deterministic = True
)

trainer.test(model, data_module)
