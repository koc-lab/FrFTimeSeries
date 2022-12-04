import tensorflow as tf
import numpy as np
#from src.ar_synthetic_data import autoregressive_syn
import pandas as pd
from src.narma import narma_generator
from src.electric_load import electric_loader


def generate_mackey(batch_size=100, tmax=200, delta_t=1, rnd=True):
    """
    Generate synthetic training data using the Mackey system
    of equations (http://www.scholarpedia.org/article/Mackey-Glass_equation):
    dx/dt = beta*(x'/(1+x'))
    The system is simulated using a forward euler scheme
    (https://en.wikipedia.org/wiki/Euler_method).

    Returns:
        spikes: A Tensor of shape [batch_size, time, 1],
    """
    with tf.variable_scope('mackey_generator'):
        steps = int(tmax/delta_t) + 100
        
        # multi-dimensional Mackey data.
        def mackey(x, tau, gamma=0.1, beta=0.2, n=10):
            return beta*x[:, -tau]/(1 + tf.pow(x[:, -tau], n)) - gamma*x[:, -1]
        # multi-dimensional autoregressive model
        def autoregressive_syn(batch_size=50, duration=4000,seed_number=0):
                ar = np.zeros((batch_size, duration))
                np.random.seed(seed_number)
                print("Create AR(1) data creation")
                for bb in range(batch_size):
                    for dd in range(1, duration):
            
                        if dd % 1000 < 500:
                            ar[bb, dd] = ar[bb, dd - 1] + np.random.normal(0, 0.01, 1)
                        elif dd % 1000 >= 500:
                            ar[bb, dd] = ar[bb, dd - 1] * -0.9 + np.random.normal(0, 0.01, 1)
                ar_min= ar.min(axis=1)[0]
                ar_max= ar.max(axis=1)[0]
                ar= (ar-ar_min)/(ar_max-ar_min)
                
                ar=tf.convert_to_tensor(ar)
                ar= tf.cast(ar,dtype=tf.float32)
                print("Autoregressive data is created")
                return ar
                
        def exchange_currency():
           file_dir= #directory of currency data
           usd_try= pd.read_csv(file_dir)
           
           # Train and test sets are different, 
           usd_try = usd_try["Close"][0:1280] # Train
           #usd_try = usd_try["Close"][1280:]   #Test
           
           utr= usd_try.to_numpy()
           nan_list = np.argwhere(np.isnan(utr))

        ######################################################## This is for normalization, it is up to you ###############################################
           for nn in nan_list:
             utr[nn[0]]=(utr[nn[0]-1]+utr[nn[0]+1])/2
           utr= (utr-np.min(utr))/(np.max(utr)-np.min(utr))
        ###################################################################################################################################################
           us_try = np.reshape(utr,(1,utr.shape[0]))
           us_try= tf.cast(us_try,dtype=tf.float32)
           print(us_try.shape)
           return us_try


        tau = int(17*(1/delta_t))
        x0 = tf.ones([tau])
        x0 = tf.stack(batch_size*[x0], axis=0)
        if rnd:
            print('Mackey initial state is random.')
            x0 += tf.random_uniform(x0.shape, -0.1, 0.1,seed=1)
        else:
            x0 += tf.random_uniform(x0.shape, -0.1, 0.1, seed=0)

        x = x0
        with tf.variable_scope("forward_euler"):
            for _ in range(steps):
                res = tf.expand_dims(x[:, -1] + delta_t*mackey(x, tau), -1)
                x = tf.concat([x, res], -1)
        discard = 100 + tau
################################################################## This part is prepared for other datasets ################################################
#            cur = exchange_currency()
#            narma_series=narma_generator(3)
#            AR= autoregressive_syn(10,5120,15)
#            electric_dataset= #directory for electric load dataset
#            electric= electric_loader(electric_dataset)
#############################################################################################################################################################
           
       
        return x[:, discard:]


if __name__ == "__main__":
    tf.enable_eager_execution()
    import matplotlib.pyplot as plt
    # import matplotlib2tikz as tikz
    mackey = generate_mackey(tmax=1200, delta_t=0.1, rnd=True)
    print(mackey.shape)
    plt.plot(mackey[0, :].numpy())
    # tikz.save('mackey.tex')
    plt.show()


class MackeyGenerator(object):
    '''
    Generates lorenz attractor data in 1 or 3d on the GPU.
    '''

    def __init__(self, batch_size, tmax, delta_t, restore_and_plot=False):
        self.batch_size = batch_size
        self.tmax = tmax
        self.delta_t = delta_t
        self.restore_and_plot = restore_and_plot

    def __call__(self):
        data_nd = generate_mackey(tmax=self.tmax, delta_t=self.delta_t,
                                  batch_size=self.batch_size,
                                  rnd=not self.restore_and_plot)
        data_nd = tf.expand_dims(data_nd, -1)
        print('data_nd_shape', data_nd.shape)
        return data_nd
