import lstmAttention2
import lstmAttention
import functions
import torch
import os
from torch.utils.data import DataLoader
import datasetClasses
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import * 
import LSTM_Temperature

device = "cuda"
#device = "cpu"

#Model Name
modelName = "LSTMAttentionWithTemperatureLSTMCNN_Little"
# args:args: lstmLayersEnc, lstmLayersDec, lstmHiddenSize, lstmInputSize, dropout, attentionHeads, device
model = lstmAttention2.LSTM(1,1, 2500, 2500, 0.1, 5,  device).to(device)
#model = lstmAttention.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)

# load weights
#model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", modelName))

# define hyperparameters
params = {"learningRate": 0.0001, "weightDecay": 0.001, "epochs": 20, "batchSize": 8, "optimizer": "adam", "validationStep": 100}



# get dataLoaders
path_images = os.path.join(path, "datasets", name, "alignedAveragedDataNDSIPatched")
path_temperatures= os.path.join(path, "datasets", name, "TemperatureDataPatched")
#datasetTrain = datasetClasses.glaciers("/home/jonas/datasets/parbati", "train")
datasetTrain = datasetClasses.glaciers(path_images, path_temperatures, "train")
dataTrain = DataLoader(datasetTrain, params["batchSize"], shuffle = True)

datasetVal = datasetClasses.glaciers(path_images, path_temperatures, "val")
#datasetVal = datasetClasses.glaciers("/home/jonas/datasets/parbati", "val")
dataVal = DataLoader(datasetVal, params["batchSize"], shuffle = True)


# criterion
#loss = torch.nn.MSELoss()
loss = torch.nn.L1Loss()

# optimizer with weight decay for regularization
optimizer = torch.optim.Adam(model.parameters(), lr=params["learningRate"], weight_decay=params["weightDecay"])

# train on patches
## args: trainLoader, valLoader, tokenizer, model, criterion, loadModel, modelName, params,  WandB, device, pathOrigin = pathOrigin
functions.trainLoop(dataTrain, dataVal,  model, loss, False, modelName, params, True, device, optimizer=optimizer)
