import numpy as np
import os
import pandas as pd
import pickle
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from data.prepareData import prepareDataLoader_fromPkl, prepareDataLoader_fromTensor
from models.autoencoder import methVAE
from scripts.train_EarlyStopping import train_VAE

def loadData(PATH_train, PATH_val, batch_size=64):
    train_dataset = prepareDataLoader_fromPkl(PATH_train)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True, 
                              num_workers=8)

    val_dataset = prepareDataLoader_fromPkl(PATH_val)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=True, 
                            num_workers=8)
    
    return train_dataset, train_loader, val_dataset, val_loader

def main():
    PATH_data = "data"
    PATH_train = os.path.join(PATH_data, args.train_file)
    PATH_val = os.path.join(PATH_data, args.val_file)
    
    ### Load Data
    train_dataset, trainLoader, val_dataset, valLoader = loadData(PATH_train, PATH_val)
    
    ### Train model   
    inputDim = train_dataset.returnTensor_()[0].shape[1]
    args.hidden_layer_encoder_topology = list(map(int,args.hidden_layer_encoder_topology))
    model = methVAE(inputDim=inputDim,
                 hidden_layer_encoder_topology=args.hidden_layer_encoder_topology,
                 latentSize=args.latentSize)
    print(model)
    train_VAE(logName=args.name,
        model=model, 
        train_loader=trainLoader, 
        val_loader=valLoader,
        criterion=nn.MSELoss(reduction="sum"),
        n_epochs=args.n_epochs, 
        lr=args.learning_rate,
        patienceEarlyStopping=10,
        sleep_earlyStopping=100)
         
    
if __name__ == "__main__":
    ## add argparse here
    parser = argparse.ArgumentParser(description='Train AE')
    parser.add_argument('--name', type=str)
    parser.add_argument('--train_file', type=str)
    parser.add_argument('--val_file', type=str)
    parser.add_argument('--hidden_layer_encoder_topology', nargs="+", default=[])
    parser.add_argument('--latentSize', type=int)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--n_epochs', type=int, default=500)
    
    args = parser.parse_args()
    main()