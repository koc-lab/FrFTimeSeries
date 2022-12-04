# Created by moritz (wolter@cs.uni-bonn.de) at 28/02/2020

import io
import copy
import time
import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from src.mackey_glass_generator import MackeyGenerator
from src.power_experiments.prediction_graph import FFTpredictionGraph
import pandas as pnd

def np_scalar_to_summary(tag: str, scalar: np.array, np_step: np.array,
                         summary_file_writer: tf.summary.FileWriter):
    """
    Adds a numpy scalar to the logfile.
    :param tag: The tensorboard plot title.
    :param scalar: The scalar value to be recordd in that plot.
    :param np_step: The x-Axis step
    :param summary_file_writer: The summary writer used to do the recording.
    """
    mse_net_summary = tf.Summary.Value(tag=tag, simple_value=scalar)
    mse_net_summary = tf.Summary(value=[mse_net_summary])
    summary_file_writer.add_summary(mse_net_summary, global_step=np_step)


def run_experiemtns(lpd_lst):
    
  
   
    for exp_no, lpd in enumerate(lpd_lst):
       
          
            total_time =0
            time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            lpd['time_str'] = time_str
           
            pgraph = FFTpredictionGraph(lpd, generator=lpd['generator'])
            param_str = time_str + '_' + lpd['cell_type'] + '_size_' + str(lpd['num_units']) + \
                '_fft_' + str(lpd['fft']) + \
                '_bs_' + str(lpd['batch_size']) + \
                '_ps_' + str(lpd['pred_samples']) + \
                '_dis_' + str(lpd['discarded_samples']) + \
                '_lr_' + str(lpd['init_learning_rate']) + \
                '_dr_' + str(lpd['decay_rate']) + \
                '_ds_' + str(lpd['decay_steps']) + \
                '_sp_' + str(lpd['sample_prob']) + \
                '_rc_' + str(lpd['use_residuals']) + \
                '_pt_' + str(pgraph.total_parameters) + '_fr_'+ str(lpd['fraction'])
    
            if lpd['fft']:
                param_str += '_wf_' + str(lpd['window_function'])
                param_str += '_ws_' + str(lpd['window_size'])
                param_str += '_ol_' + str(lpd['overlap'])
                param_str += '_ffts_' + str(lpd['step_size'])
                param_str += '_fftp_' + str(lpd['fft_pred_samples'])
                param_str += '_fl_' + str(lpd['freq_loss'])
                param_str += '_eps_' + str(lpd['epsilon'])
                param_str += '_fftcr_' + str(lpd['fft_compression_rate'])
    
            if lpd['downsampling'] > 1:
                param_str += '_downs_' + str(lpd['downsampling'])
    
            if lpd['stiefel']:
                param_str += '_stfl'
    
            if lpd['linear_reshape']:
                param_str += '_linre'
    
            print('---------- Experiment', exp_no, 'of', len(lpd_lst), '----------')
            print(param_str)
            # print(lpd)
            summary_writer = tf.summary.FileWriter(lpd['base_dir'] + param_str, graph=pgraph.graph)
             #dump the parameters
            with open(lpd['base_dir'] + param_str + '/param.pkl', 'wb') as file:
                pickle.dump(lpd, file)
    
            gpu_options = tf.GPUOptions(visible_device_list=str(lpd['GPUs'])[1:-1])
            #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
            config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False,
                                    gpu_options=gpu_options)
            with tf.Session(graph=pgraph.graph, config=config) as sess:
              pgraph.init_op.run()
              for it in range(lpd['iterations']):
                  
               
                  start = time.time()
                  # array_split add elements here and there, the true data is at 1
                  if not lpd['fft']:
                      np_loss, summary_to_file, np_global_step, _, datdec_np, decout_np, \
                          datand_np = \
                          sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                                    pgraph.training_op, pgraph.data_decoder_time,
                                    pgraph.decoder_out, pgraph.data_nd])
                  else:
                      
                      np_loss, summary_to_file, np_global_step, _, datdec_np, decout_np, \
                          datand_np, window_np = \
                          sess.run([pgraph.loss, pgraph.summary_sum, pgraph.global_step,
                                    pgraph.training_op, pgraph.data_decoder_time,
                                    pgraph.decoder_out, pgraph.data_nd, pgraph.window])
                  stop = time.time()
          
                  if it % 25 == 0:
                      print('it: %5d, loss: %5.6f, time: %1.2f [s]'
                            % (it, np_loss, stop-start))
                  # debug_here()
                  summary_writer.add_summary(summary_to_file, global_step=np_global_step)
                  np_scalar_to_summary('runtime', stop-start, np_global_step, summary_writer)
                  total_time = total_time + stop-start
               
        
              pgraph.saver.save(sess, lpd['base_dir'] + param_str +'/weights/cpk',global_step=np_global_step)
          
              print('wow')
    
     
