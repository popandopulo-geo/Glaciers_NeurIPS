import pandas as pd
import functions
import torch
import os
from torch.utils.data import DataLoader
import datasetClasses
import numpy as np
import matplotlib.pyplot as plt
#import transformerBase
import LSTM
import ConvLSTM
import lstmAttention
#from unet_model import UNet
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from config import * 

from DeepLearning.LSTMTrain_temperature.datasetClasses import glaciers as gl
import DeepLearning.LSTMTrain_temperature.lstmAttention2 as lstmAttention2_temp
## global variables for project
### change here to run on cluster ####
pathOrigin = path
#pathOrigin = "C:\\FAU\\3rd SEM\\Glaciers_NeurIPS"
#pathOrigin = "/media/jonas/B41ED7D91ED792AA/Arbeit_und_Studium/Kognitionswissenschaft/Semester_5/masterarbeit#/data_Code"
device = "cuda"
tokenizer = False

if tokenizer:
    # load tokenizer and model
    Tokenizer = tokenizer.tokenizer()
    os.chdir(pathOrigin + "/models")
    Tokenizer = functions.loadCheckpoint(Tokenizer, None, pathOrigin + "/models/" + "tokenizer")
    Tokenizer = Tokenizer.to(device)

modelName = "LSTMAttentionWithTemperatureLSTMCNN_Little"
#model = ConvLSTM.ConvLSTMPredictor([64, 64, 24, 24, 64, 24]).to(device)
model = lstmAttention2_temp.LSTM(1,1, 2500, 2500, 0.1, 5,  device).to(device)
#model = LSTM.LSTM(3,3, 2500, 2500, 0.1, 5,  device).to(device)
#model = UNet(1,1).to(device)


# load weights to transformers
model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", modelName))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "Unet"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "ConvLSTM"))
# model = functions.loadCheckpoint(model, None, os.path.join(pathOrigin, "models", "LSTMEncDec"))
print("loading models finished")

# get dataLoaders
path_images = os.path.join(pathOrigin, "datasets", name, "alignedAveragedDataNDSIPatched")
path_temperatures= os.path.join(pathOrigin, "datasets", name, "TemperatureDataPatched")
# dataLoader /home/jonas/datasets/parbati
datasetTest = gl(path_images, path_temperatures, "val")
#datasetTest = datasetClasses.glaciers("/home/jonas/datasets/parbati", "test", bootstrap = True)
dataTest = DataLoader(datasetTest,256,  shuffle = False)


with torch.no_grad():
    # do 2000 bootstrap iterations
    losses = []
    nIterations = 1
    MSELoss = torch.nn.MSELoss()
    MAELoss = torch.nn.L1Loss()
    modelResults = np.zeros((nIterations, 2))

    for b in range(nIterations):
        MSElosses = torch.zeros(len(dataTest))
        MAElosses = torch.zeros(len(dataTest))
        counter = 0

        # check model performance on bootstrapped testset
        for inpts, targets , temps, idx in dataTest:

            inpts = inpts.to(device).float()
            targets = targets.to(device).float().squeeze()
            temperatures=[]
            for t in temps:
                temperatures.append(t.to(device).float())

            if tokenizer:
                # encode with tokenizer and put to gpu
                inpts = functions.tokenizerBatch(Tokenizer, inpts, "encoding", device)
                targets = functions.tokenizerBatch(Tokenizer, targets, "encoding", device)

            # predict
            model.eval()
            forward = model.forward(inpts, temperatures, targets, training = False)

            if tokenizer:
                forward = Tokenizer.decoder(forward)
                forward = functions.tokenizerBatch(Tokenizer, forward, "decoding", device)
                forward = torch.reshape(forward, (1, forward.size(0), 50, 50))

            
            if inpts.size(0) <0:
                for i in range(forward.size(0)):
                    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                    # plot numpy arrays forward and target next to each other in the same plot in gray scale

                    axs[0].imshow(forward[i,0 ,:, :].cpu().numpy(), cmap='gray')
                    axs[0].set_title('Predicted')
                    axs[1].imshow(targets[i,0, :, :].cpu().numpy(), cmap='gray')
                    axs[1].set_title('Target')
                    #save the plt plot to a jpeg file without displaying it
                    os.chdir(os.path.join(pathOrigin, "prints",modelName))
                    plt.savefig("nIteration" + str(b) + "_" + str(counter) + "_" + str(i)+ ".jpeg")
                    plt.close()

            

            # get loss
            MSE = MSELoss(forward, targets)
            MAE = MAELoss(forward, targets)
            MSElosses[counter] = MSE
            MAElosses[counter] = MAE
            counter += 1

            # convert forward and targets to numpy arrays and 
            # resize to torch.Size([4, 800, 800])
            forward = forward.cpu().numpy()
            targets = targets.cpu().numpy()

            #put first block of 4x50x50 at the final position
            # roll = 10
            # forward = np.roll(forward, roll, axis=0)
            # targets = np.roll(targets, roll, axis=0)
            idx = [int(i) for i in idx]
            _forward = np.zeros((256,4,50,50))
            _targets = np.zeros((256,4,50,50))
            for i in range(len(forward)):
                _forward[idx[i]] = forward[i]
                _targets[idx[i]] = targets[i]

            f = np.zeros((4, 800, 800))
            for i in range(16):
                for j in range(16):
                    f[:, 50*j:50*(j+1), 50*i:50*(i+1)] = _forward[j+16*i, :, :, :]


            #vecto of 256x4x50x50, shoud be 16x4x800x50, rearranging blocks of 50x50 to 800x800
            z = np.zeros((4, 800, 800))
            for i in range(16):
                for j in range(16):
                    z[:, 50*j:50*(j+1), 50*i:50*(i+1)] = _targets[j+16*i, :, :, :]

            #save in numpy array as an image
            for i in range(f.shape[0]):
                pred = f[i]
                tar = z[i]
                #save numpy array as an image
                plt.imsave(os.path.join(pathOrigin, "prints",modelName, "Prediction_" + str(i)+ ".jpeg"), pred, cmap='gray')
                plt.imsave(os.path.join(pathOrigin, "prints",modelName, "Target_" + str(i)+ ".jpeg"), tar, cmap='gray')

            fig, axs = plt.subplots(2, 4, figsize=(25, 15))
            # plot numpy arrays forward and target next to each other in the same plot in gray scale
            for i in range(f.shape[0]):
                axs[0, i].imshow(f[i, :, :], cmap='gray')
                axs[0, i].set_title('Predicted ' + str(i))
                #axis off
                axs[0, i].axis('off')
                axs[1, i].imshow(z[i, :, :], cmap='gray')
                axs[1, i].set_title('Target ' + str(i))
                #axis off
                axs[1, i].axis('off')
            #save the plt plot to a jpeg file without displaying it
            os.chdir(os.path.join(pathOrigin, "prints",modelName))
            plt.savefig("Plot_total.jpeg")
            plt.close()

            #
        MSE = torch.mean(MSElosses)
        MAE = torch.mean(MAElosses)
        modelResults[b, 0] = MSE.detach().cpu().numpy()
        modelResults[b, 1] = MAE.detach().cpu().numpy()

    # save model results
    df = pd.DataFrame(modelResults, columns = ["MSE", "MAE"])
    os.chdir(os.path.join(pathOrigin, "models"))
    df.to_csv(modelName + "_bootstraped_results" + ".csv")
    os.chdir(pathOrigin)
    print("Test model results finished")