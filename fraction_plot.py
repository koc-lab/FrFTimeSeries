
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os




'''
Reading from list of model files of paths and, print the final fraction order of the experiment in a list and plot and save the change of fraction order for each experiment

'''
#fig_dir= directiort to save figures
#source_path = directory that contains many experiment
#file_lst = []
#for root, dirs, files in os.walk(source_path):
#  if not root.split('/')[-1] == 'weights':
#            #print(dirs)
#            #print(files)
#            for file in files:
#                if file.split('.')[0] == 'events':
#                    file_lst.append(os.path.join(root, file))
#
#
#counter = 0
#learned_frac_list= []
#total_iter=100
#for event_path in file_lst:
# 
#    counter=counter+1
#    iterit=0
#    value_list= []
#    for e in tf.compat.v1.train.summary_iterator(event_path):
#        for v in e.summary.value:
#            if v.tag == 'fraction_order/fraction':
#                iterit=iterit+1
#                value_list.append(v.simple_value)
#                if iterit==total_iter:
#                   learned_frac_list.append(v.simple_value)
#                #print(v.simple_value)
#                #print("Iteration", iterit)
#    x_ax=np.arange(1,iterit+1)
#    plt.title(" Iteration vs Fraction Order (a)" )
#    plt.xlabel('Iterations')
#    plt.ylabel('a')
#    plt.figure()
#    plt.plot(x_ax,value_list)
#  
#    fig_name= #figure names
#    plt.savefig(fig_dir+fig_name)
#
#
#
#with open("fraction_list.txt", "w") as output:
#    output.write(str(learned_frac_list))



'''
Only one experiment. Print and plot the fraction orders
'''


iterit=0
value_list= []    
event_path= # path of the events in a path
for e in tf.compat.v1.train.summary_iterator(event_path):
       for v in e.summary.value:
            if v.tag == 'fraction_order/fraction':
                iterit=iterit+1
                value_list.append(v.simple_value)
                print(v.simple_value)
                print("Iteration", iterit)
x_ax=np.arange(1,iterit+1)
plt.title(" Iteration vs Fraction Order (a)" )
plt.xlabel('Iterations')
plt.ylabel('a')
plt.plot(x_ax,value_list)
  
fig_name= #figure name
plt.savefig(fig_name)            
                    
                
    
