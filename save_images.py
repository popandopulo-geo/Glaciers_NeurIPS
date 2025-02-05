import pickle
import os

path = "C:\\FAU\\3rd SEM\\Glaciers_NeurIPS\datasets\parvati"

# path concatenated with 'TemperatureData'
path_temperatures = os.path.join(path, "TemperatureData")

#load data pickle file '0'
file = os.path.join(path_temperatures, '0')
with open(file, "rb") as f:
    data = pickle.load(f)

#print(type(data))
#print(data.shape)


#plot first 2 images from data in one plot and then save it

import matplotlib.pyplot as plt
images = data[0:2]
fig, ax = plt.subplots(1, 2)
#do not plot axis
ax[0].axis("off")
ax[1].axis("off")
ax[0].imshow(images[0])
ax[1].imshow(images[1])
#title of the first image is 'Temperature in daylight' and second image is 'Temperature in night'
ax[0].set_title("Temperature in daylight")
ax[1].set_title("Temperature in night")
plt.savefig(os.path.join(path, "images", "iamge.png"))

path_temp_patches = os.path.join(path, "TemperatureDataPatched","images")

#load data pickle file '0'
file = os.path.join(path_temp_patches, '0')
with open(file, "rb") as f:
    data = pickle.load(f)

print(type(data))
print(data[1].shape)
data = data[1]
#plot first 8 images from data in one plot and then save it
images = data[0:12]
fig, ax = plt.subplots(3, 4)
#do not plot axis  
for i in range(3):
    for j in range(4):
        ax[i, j].axis("off")
        ax[i, j].imshow(images[i*4+j])
#title of the plot is "Temperature patches"
fig.suptitle("Temperature patches")
plt.savefig(os.path.join(path, "images", "patches.png"))
