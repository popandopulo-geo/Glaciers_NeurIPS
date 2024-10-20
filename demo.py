import os
import pickle
import planetary_computer as pc
from numpy import array
import copy
from collections import Counter
import stackstac
import numpy as np
import pystac_client
import rioxarray
from scipy.ndimage import zoom
import pandas as pd

# configuration for data extraction
path = "C:\\FAU\\3rd SEM\\Glaciers_NeurIPS" # file path to root folder

years = ["2013"]
boundingBox = (77.4879532598204, 31.992646374004327, 77.82063300835557, 32.225292222190056)
extractNDSI = True # False is RGB bands, read documentatiojn if all/different bands of satellite are necessary
clouds = 20 # 0 - 100
extractedCoordinates = [0, 800, 0, 800] # extracted pixels from the satellite images 
allowedMissings = 0.5 # 0 - 1.0 
deltaT = 6 # average data over deltaT months
name = "parvati_temp"
# ECC maximization parameters
stoppingCriterion = 1e-4
maxIterations = 10000
# if patches are extracted 
patchSize = 50
stride = 10
sequenceLength = 4

def interpolate_expand(vector):
    """
    Expands and interpolates a NumPy array into a larger array.

    Parameters
    ----------
    vector: numpy.ndarray
        A array of shape (d, height, width) containing input data with potential NaN values.
        - d: number of time steps or observations.
        - height: original height of each observation.
        - width: original width of each observation.

    Returns
    -------
    numpy.ndarray
        A 3D array of shape (d, 800, 800) where each original 2D slice has been interpolated and resized.
    """

    d,_,_ = vector.shape
    data2 = np.zeros((vector.shape[0],800,800))
    target_shape = (data2.shape[1],data2.shape[2])
    for i in range(d):
        data = vector[i]

        # Convert the array to a DataFrame for easier manipulation
        df = pd.DataFrame(data)

        # Interpolate NaN values based on their surroundings
        df_interpolated = df.interpolate(method='linear', axis=1).interpolate(method='linear', axis=0)

        # Convert back to NumPy array if needed
        data = df_interpolated.to_numpy()

        # Calculate the zoom factors for each axis
        zoom_factors = (target_shape[0] / data.shape[0], target_shape[1] / data.shape[1])

        # Use zoom to resize and interpolate the array
        expanded_array = zoom(data, zoom_factors, order=3)

        # Truncate to two decimals
        expanded_array = np.trunc(expanded_array * 100) / 100
        data2[i] = expanded_array

    return data2


def getData(bbox, bands, timeRange, cloudCoverage, allowedMissings):
    """
    gets data in numpy format

    bbox: list of float
        rectangle to be printed
    bands: list of string
        ['QC_Day', 'Emis_31', 'Emis_32', 'QC_Night', 'LST_Day_1km', 
        'Day_view_angl', 'Day_view_time', 'LST_Night_1km', 
        'Clear_sky_days', 'Night_view_angl', 
        'Night_view_time', 'Clear_sky_nights']
    timeRange: string
        e.g. "2020-12-01/2020-12-31"
    cloudCoverage: int
        amount of clouds allowed in [0,100]
    allowedMissings: float
        amount of pixels nan

    returns: list of tuple of datetime array and 4d numpy array and cloudCoverage array
        [time, bands, x, y]

    """

    catalog = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')

    # query
    search = catalog.search(
        collections=['modis-11A2-061'],
        max_items= None,
        bbox=bbox,
        datetime=timeRange
    )

    items = pc.sign(search)
    print("found ", len(items), " images")

    # stack
    stack = stackstac.stack(items.items, bounds_latlon=bbox, epsg=4326)

    # use common_name for bands
    stack = stack.assign_coords(band=stack.band.rename("band"))
    output = stack.sel(band=bands)

    # put into dataStructure
    t = output.shape[0]
    
    cloud_cover = np.array(output["eo:cloud_cover"], dtype=float)
    cloud_cover[np.isnan(cloud_cover)] = 0
    cloud = np.array(cloud_cover <= cloudCoverage)
    cloud = [cloud, np.array(output["eo:cloud_cover"])]
    time = np.array(output["start_datetime"])
    
    # Get dimension names
    print("Dimension names:", output.dims)

    # Get coordinate names
    print("Coordinates:", output.coords)
    print("proj:geometry", output["x"])
    print("proj:geometry", output["y"])

    dataList = []
    for i in range(t):
        if cloud[0][i] == True:  # check for clouds
            if np.count_nonzero(np.isnan(output[i, 1, :, :])) >= round(
                (output.shape[2] * output.shape[3]) * allowedMissings):  # check for many nans
                pass
            elif np.count_nonzero(np.isnan(output[i, 1, :, :])) <= round(
                        (output.shape[2] * output.shape[3]) * allowedMissings):
                    data = array([np.array(output[i, 0, :, :]), # get seven bands of the satellite sensors
                                np.array(output[i, 1, :, :]),
                                np.array(output[i, 2, :, :]),
                                np.array(output[i, 3, :, :])
                                ])
                    data = interpolate_expand(data)
                    cloudCov = cloud[1][i]
                    print(time[i])
                    data = (time[i], data, cloudCov)
                    dataList.append(data)
    print("done!")
    dat= dataList[0][1]
    print(type(dat))
    print(dat.shape)    
    print(dat[2])
    return dataList

# wrapper for acquiring data
def API(box, 
        time, 
        cloudCoverage, 
        allowedMissings, 
        year, 
        glacierName, 
        bands = ['Emis_31', 'Emis_32', 'LST_Day_1km', 'LST_Night_1km']):
    """
    acquire and preprocess the data

    box: tuple of float
        coordinate box from which images are taken
    time: string
        time range for extraction of data
    cloudCoverage: int
        percent of pixels covered with clouds in bands
    allowedMissings: float
        p(missingData)
    year: string
        year of data extraction, downloads chunks of data as one year packages
    glacierName: string
        Name of the glacier for folder structure

    return: list of tuple of datetime and 4d ndarray tensor for model building
    """
    # get data
    d = getData(bbox=box, bands=bands, timeRange=time, cloudCoverage= cloudCoverage, allowedMissings=allowedMissings)

    # save on hard drive with pickling
    # create folder
    pathOrigin = path + "/" + "datasets" + "/" + glacierName + "/" + "rawData"
    os.makedirs(pathOrigin, exist_ok = True)
    os.chdir(pathOrigin)

    # save data object
    with open(year, "wb") as fp:  # Pickling
        pickle.dump(d, fp)
    print("data saved!")
    return d

def getYearlyData(years, 
                  boundingBox, 
                  clouds, 
                  allowedMissings, 
                  name):
    """
    years: list of string
        years that are to be extracted
    boundingBox: tuple of float
        tuple with four coordinates for the scene boundary
    clouds: int 
        int [0,100]
    allowedMissings: float
        [0,1], amount of missing pixels allowed
    Name: str
        Name of folder
    """

    for b in range(len(years)): #- 1):
        os.chdir(path)
        if b < 10: ## fix here !!
            print("#####################################################################################################")
            print("start processing year: " + years[b])
            #string = years[b] + "-01-01/" + years[b+1] + "-01-01"
            string = years[b] + "-01-01/" + str(int(years[b])) + "-02-01"
            print(string)
            API(boundingBox,
                string,
                clouds,
                allowedMissings, 
                years[b],
                name)
            
            print(years[b] + " done")
    
    return None

if __name__ == '__main__':
    getYearlyData(years, boundingBox, clouds, allowedMissings, name)