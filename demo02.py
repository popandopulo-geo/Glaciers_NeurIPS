import numpy as np
import pandas as pd
from scipy.ndimage import zoom

def interpolate_expand(vector):
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
        expanded_array = np.trunc(expanded_array * 100) / 100
        data2[i] = expanded_array
    return data2


# Loading the array back
data = np.load('data_example.npy', allow_pickle=True)
data[data==-50.] = np.nan

data = interpolate_expand(data)
print(data)


x = np.ones((4,29,19))
x = interpolate_expand(x)
print(x)