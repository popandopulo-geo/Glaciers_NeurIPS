import os
from torch.utils.data import Dataset
from torch import is_tensor
import functions
from torch.utils.data import DataLoader
import numpy as np

class glaciers(Dataset):
    def __init__(self, path, path_temperatures, mode):
        """
        dataset class for train loop
        path: str
            path to image and target folder
        mode: str
            train, val, test
        """
        self.mode = mode
        self.path = path
        self.path_temperatures = path_temperatures

        # take 80% of data as train and val, take 20% of data for testing
        # get list of all image paths in directory
        images = os.listdir(os.path.join(self.path, "images"))
        paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
        # take 80% of data as training
        criterion = round(len(paths) * 0.8) # turn off for aletsch as is already splitted, change for 0.8
        #change to one because i separate the test data to 2018 and 2019
        criterion = round(len(paths) * 1)
        paths = paths[0:criterion]
        self.codigo = images
        self.images = paths

        targets = os.listdir(os.path.join(self.path, "targets"))
        paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
        paths = paths[0:criterion]
        self.targets = paths

         # Load temperature data similarly
        temperatures = os.listdir(os.path.join(self.path_temperatures, "images"))
        paths = [os.path.join(os.path.join(self.path_temperatures, "images"), item) for item in temperatures]
        paths = paths[0:criterion]
        self.temperatures = paths

        if self.mode == "train":
            # take 90% of data as training
            criterion = round(len(self.images) * 0.90)
            self.images = self.images[0:criterion]
            self.targets = self.targets[0:criterion]
            self.temperatures = self.temperatures[0:criterion]

        if self.mode == "val":
            # take 10% as validation set
            criterion = round(len(paths) * 0.9) # cero because i want to test all
            self.images = self.images[criterion:]
            self.targets = self.targets[criterion:]
            self.temperatures = self.temperatures[criterion:]

        if self.mode == "test":
            # get list of all image paths in directory
            images = os.listdir(os.path.join(self.path, "images"))
            paths = [os.path.join(os.path.join(self.path, "images"), item) for item in images]
            criterion = round(len(paths) * 0.8)
            paths = paths[criterion:]
            self.images = paths

            targets = os.listdir(os.path.join(self.path, "targets"))
            paths = [os.path.join(os.path.join(self.path, "targets"), item) for item in targets]
            paths = paths[criterion:]
            self.targets = paths

            temperatures = os.listdir(os.path.join(self.path_temperatures, "images"))
            paths = [os.path.join(os.path.join(self.path, "images"), item) for item in temperatures]
            paths = paths[criterion:]
            self.temperatures = paths


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        returns datum for training

        idx: int
            index to datum
        returns: torch.tensor
                image and targets
        """
        if is_tensor(idx):
            idx = idx.tolist()

        try:
            # get data in tensor format
            inpt = functions.openData(self.images[idx])
            #inpt = inpt[:, 2, :, :]
            target = functions.openData(self.targets[idx])
            temperature = functions.openData(self.temperatures[idx])  # load temperature data
        except:
            # get data in tensor format
            index = np.random.randint(self.__len__())
            inpt = functions.openData(self.images[index])
            #inpt = inpt[:, 2, :, :]
            target = functions.openData(self.targets[index])
            temperature = functions.openData(self.temperatures[index])

        return inpt, target, temperature, self.codigo[idx]
