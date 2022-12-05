#  Fractional Fourier Transform in Time Series Prediction

This is the GitHub repository for the paper: E.Koc,  A. Koç, **“ Fractional Fourier Transform in Time Series Prediction ”** submitted to IEEE Signal Processing Letters, 2022. In this study, we apply a window function to a univariate time series and divide it into segment. Then, we apply fractional Fourier transform on each segment and obtain feature vectors. These feature vectors are fed into GRU/RNN based encoder-decoder to predict the future time series. This repository is benefited from the codes in the paper [Sequence Prediction using Spectral RNNs](https://github.com/v0lta/Spectral-RNN) and modified the codes depending on the tasks. Codes are implemented using Tensorflow 1.15 (1.1x is also suitable )on Linux based systems 

 #### Training ####
Run the script `python3 frft_test.py` to train the network. There are several important points to modify in `frft_test.py` for future studies. `pd['base_dir']` is the directory that you save your models. You can run the list of multiple experiments and save the models here. 


 #### Test ####
 
 To test the models, run the script `python3 mse_test.py`. 
 
 ## References
 
M. Wolter, J. Gall, and A. Yao, “Sequence prediction using spectral RNNs,”
in International Conference on Artificial Neural Networks. Springer, 2020,
pp. 825–837.
