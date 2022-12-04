import io
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.mackey_glass_generator import MackeyGenerator
from src.power_experiments.prediction_graph import FFTpredictionGraph
import tikzplotlib as tikz
import os
import pandas as pnd
import tensorflow as tf

'''
After the training of models, you should test the models use mse_test.py. Prior to run this code, go to the mackey_glass_generator or electric_load, change the options of datasets from training
to test.
'''


def plot(path, restore_step, label, gt=False):
    pd = pickle.load(open(path + '/param.pkl', 'rb'))
    mackeygen = MackeyGenerator(pd['batch_size'],
                                pd['tmax'], pd['delta_t'],
                                restore_and_plot=True)
    pgraph = FFTpredictionGraph(pd, mackeygen)
    
    print('one')

    # plot this.
    gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    print('two')

    with tf.Session(graph=pgraph.graph, config=config) as sess:
      pgraph.saver.restore(sess, save_path=path
                           + '/weights/' + 'cpk' + '-' + str(restore_step))
      if not pd['fft']:
        np_loss, summary_to_file, np_global_step, \
            datenc_np, datdec_np, decout_np, \
            datand_np = \
            sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                      pgraph.data_encoder_time, pgraph.data_decoder_time,
                      pgraph.decoder_out, pgraph.data_nd])
      else:
        np_loss, summary_to_file, np_global_step, \
            datenc_np, datdec_np, decout_np, \
            datand_np, window_np = \
            sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                      pgraph.data_encoder_time, pgraph.data_decoder_time,
                      pgraph.decoder_out, pgraph.data_nd,
                      pgraph.window])

    #print(" ******************** TEST LOSS IS*******************", np_loss)
    
    return np_loss


restore_step = # How many epoch you trained the network

source_path = # Directory of the models you saved for a specific dataset
models = sorted(os.listdir(source_path)) # 


    
filename=  # name of the file to save the test loss
loss_file = open(filename, "w")

for model_no, model in enumerate(models):

        model_path= source_path+'/'+ model
       
        loss= plot(model_path,restore_step,label="results")
        print("********************************** LOSS ****************************************",loss)
        loss_file.write(str(loss) +"\n")
        


loss_file.close()

   
   















