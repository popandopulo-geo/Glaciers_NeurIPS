import ConvLSTM
import functions
import torch
import os
from torch.utils.data import DataLoader
import datasetClasses
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config_old import *  

device = "cuda"

#device = "cpu"
#BIG
model = ConvLSTM.ConvLSTMPredictor([128, 64, 32, 32, 64, 32]).to(device)

#little
#model = ConvLSTM.ConvLSTMPredictor([2, 2, 2, 2, 2, 2]).to(device)

# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 20, "batchSize": 8, "optimizer": "adam", "validationStep": 100}


# get dataLoaders
#datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
datasetTrain = datasetClasses.glaciers(os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched"), "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(os.path.join(path, "datasets" , name, "alignedAveragedDataNDSIPatched"), "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)


# criterion
loss = torch.nn.MSELoss()

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoop(dataTrain, dataVal,  model, loss, False, "ConvLSTMBig", params, True, device)
