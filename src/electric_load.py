import pandas as pd
import numpy as np
import tensorflow as tf

def electric_loader(file_name):
    data= pd.read_csv(file_name)
    roi = data.iloc[128:128*7,:].to_numpy()
    batched_data= np.reshape(roi,(roi.shape[1],roi.shape[0]))
    batched_data = tf.convert_to_tensor(batched_data)
    batched_data=tf.cast(batched_data,dtype="float32")
    return batched_data