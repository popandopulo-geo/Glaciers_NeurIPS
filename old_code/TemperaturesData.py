from config_old import *
import dataAPI
import os
import createPatches
import glob
import alignment
from datetime import datetime
import numpy as np
import pandas as pd
import pickle

def groupdata():
    # load data
    currentPath = os.path.join(path, "datasets", name, "rawData","Temperatures")
    os.chdir(currentPath)
    files = glob.glob(os.path.join(currentPath, '*'))
    data_ = []
    
    for i in range(len(files)):
        flaged_nan = []
        data_clean = []
        data = alignment.openData(files[i])

        for j in range(len(data)):
            dataj = data[j]
            time = datetime.strptime(dataj[0], '%Y-%m-%dT%H:%M:%SZ')
            data[j] = [time,dataj[1]]

            if np.isnan(data[j][1]).any():
                flaged_nan.append(time)

        year = data[10][0].year

        for j in range(len(data)):
            dataj = data[j]
            if (dataj[0] in flaged_nan) or (dataj[0].year!=year):
                continue
            else:
                data_clean.append(dataj)
        data_.append(data_clean)
    print("Temperature data loaded")

    data = []
    for i in range(len(data_)):
        datai = data_[i]
        year = datai[5][0].year
        monthly_data = {month: [] for month in range(1, 13)}

        # Iterate over the data
        for j in range(len(datai)):
            month = datai[j][0].month  # Get the month from the datetime
            monthly_data[month].append(datai[j][1])  # Append the array to the corresponding month

        # Now stack the arrays for each month
        data3 = []
        for month in monthly_data:
            if monthly_data[month]:  # Check if there are arrays for that month
                concatenated_array = np.concatenate(monthly_data[month], axis=0)
                data3.append([month,concatenated_array])

        data.append([year, data3])
    print("Temperature data grouped by months")

    #Make groups of Delta months
    dataGroup = []
    for i in range(len(data)):
        # filter and average over months in year
        listOfList = []
        listOfListInd = []
        delta = deltaT
        year = data[i][0]
        info = data[i][1]

        for i in range(1, 12, delta): # deltas
            monthsDelta = []
            monthsDeltaInd = []
            monthsData = []

            for b in range(len(info)):
                month = info[b][0]
                if (month < i + delta) and (month >= i):
                    monthsDelta.append(month)
                    monthsDeltaInd.append(b)
                    monthsData.append(info[b][1])

            monthsData = np.concatenate(monthsData, axis=0)
            listOfList.append(monthsDelta)
            listOfListInd.append(monthsDeltaInd)
            dataGroup.append([year,monthsDelta,monthsData])
    print("Temperature data grouped by delta-months")

    for i in range(len(dataGroup)):
        os.makedirs(os.path.join(path, "datasets", name, "TemperatureData"), exist_ok=True)
        os.chdir(os.path.join(path, "datasets", name, "TemperatureData"))
        with open(str(i), "wb") as fp:
            pickle.dump(dataGroup[i][2], fp)
        


    # create patches
    #d = createPatches.automatePatching(data, patchSize, stride)
    print("Temperature data patched")

    # put into tensors
    #d = createPatches.getTrainTest(d, sequenceLength, 0,0)
    #print("Temperature data converted to tensors and saved")

    return None 

def automatePatching(data, patchSize, stride):
    """
    creates image patches sampled from a region of interest

    data: list of np.array
    patchSize: int
        size of patches
    stride: int
        moving window regulation factor

    returns: list of list of np.array
        list of list of patches from each image 
        
    """

    res = []
    for i in range(len(data)):
        print("patching scene: ", i)
        
        patches = createPatches.createPatches(data[i], patchSize, stride)
        res.append(patches)

    return res

if __name__ == "__main__":
    #For Getting Images
    bands = ['LST_Day_1km', 'LST_Night_1km']
    dataAPI.getYearlyData(years, boundingBox, clouds, allowedMissings, name, bands, "Temperatures")
    groupdata()


    # load data
    currentPath = os.path.join(path, "datasets", name, "TemperatureData")
    os.chdir(currentPath)
    files = glob.glob(os.path.join(currentPath, '*'))
    data = []
    for i in range(len(files)):
        data.append(alignment.openData(files[i]))
    print("data loaded")

    # create patches
    d = automatePatching(data, patchSize, stride)
    print("data patched")

    
    """# Write list to a binary file using pickle
    td = [d[0][0:20], d[1][0:20], d[2][0:20], d[3][0:20], d[4][0:20], d[5][0:20], d[6][0:20], d[7][0:20] ]
    with open("d.pkl", "wb") as file:
        pickle.dump(td, file)"""

    # put into tensors
    d = createPatches.getTrainTemperatures(d, sequenceLength, 0,0)
    print("data converted to tensors and saved")
    