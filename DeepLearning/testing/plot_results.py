import torch
import functions
import datasetClasses
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import * 
import DeepLearning.LSTMTrain_temperature.lstmAttention2 as lstmAttention2_temp
import DeepLearning.LSTMTrain_temperature.lstmAttention as lstmAttention_temp
import DeepLearning.LSTMTrain.lstmAttention as lstmAttention
import DeepLearning.convLSTMTrain.ConvLSTM as ConvLSTM
import numpy as np
# Load model
pathOrigin = path
device = "cuda"

model = ConvLSTM.ConvLSTMPredictor([2, 2, 2, 2, 2, 2]).to(device)
#model = ConvLSTM.ConvLSTMPredictor([128, 64, 32, 32, 64, 32]).to(device)
#model = lstmAttention.LSTM(1, 1, 2500, 2500, 0.1, 5, device).to(device)
#model = lstmAttention_temp.LSTM(1, 1, 2500, 2500, 0.1, 5, device).to(device)
#model = lstmAttention2_temp.LSTM(1, 1, 2500, 2500, 0.1, 5, device).to(device)

modelName = "ConvLSTMLittle"
model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", modelName))
print("loading models finished")

# Load data
data = [[[],[]],[[],[]]]
# get dataLoaders
path_images = os.path.join(pathOrigin, "datasets", name, "alignedAveragedDataNDSI")
#list all file in path_images
files = os.listdir(path_images)
dates = []
images = []
for i in range(4):
    image = functions.openData(os.path.join(path_images, str(i)))
    image = image.reshape(1,800,800)
    images.append(image)
    dates.append(i)
#Convert list images to a numpy array
data[0][0] = np.array(images)
data[0][1] = np.array(dates)


dates = []
images = []
for i in range(4,8):
    image = functions.openData(os.path.join(path_images, str(i)))
    #reshape image to (1,800,800) from (800,800)
    image = image.reshape(1,800,800)
    images.append(image)
    dates.append(i)
#Convert list images to a numpy array
data[1][0] = np.array(images)
data[1][1] = np.array(dates)

#numpy attay to torch tensor
data[0][0] = torch.from_numpy(data[0][0])
data[0][1] = torch.from_numpy(data[0][1])
data[1][0] = torch.from_numpy(data[1][0])
data[1][1] = torch.from_numpy(data[1][1])
# Define parameters
outputDimensions = (800, 800)
glacierName = name
predictionName = "test_prediction"

# Run inference
functions.inferenceScenes(model, data, patchSize, stride, outputDimensions, glacierName, predictionName, modelName, device, plot=True, safe=True)
