import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def autoregressive_syn(batch_size=50, duration=4000):
    ar = np.zeros((batch_size, duration, 1))

    print("Create AR(1) data creation")
    for bb in range(batch_size):
        for dd in range(1, duration):

            if dd % 1000 < 500:
                ar[bb, dd, 0] = ar[bb, dd - 1, 0] + np.random.normal(0, 0.01, 1)
            elif dd % 1000 >= 500:
                ar[bb, dd, 0] = ar[bb, dd - 1, 0] * -0.9 + np.random.normal(0, 0.01, 1)
    ar=tf.convert_to_tensor(ar)
    ar= tf.cast(ar,dtype=tf.float32)
    print("Autoregressive data is created")
    return ar

