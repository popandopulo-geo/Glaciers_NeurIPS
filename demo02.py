from config import *
import dataAPI
import os
import createPatches
import glob
import alignment
from datetime import datetime
import numpy as np
import pandas as pd
import pickle


# Read the list from the pickle file
with open(r'C:\FAU\3rd SEM\Glaciers_NeurIPS\datasets\parvati\d.pkl', "rb") as file:
    d = pickle.load(file)

print('Longitud de muestras: ', len(d))

for i in range(len(d)):
    print(d[i].shape)
print('----------------------------------')

# put into tensors
d = createPatches.getTrainTemperatures(d, sequenceLength, 0,0)
print("data converted to tensors and saved")