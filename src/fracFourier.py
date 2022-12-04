#"""
#Module to calculate the fast fractional fourier transform.
#
#"""
#
#from __future__ import division
#import numpy as np
#import scipy
#import scipy.signal as scisig
#
#
#def frft(f, a):
#    """
#    Calculate the fast fractional fourier transform.
#
#    Parameters
#    ----------
#    f : numpy array
#        The signal to be transformed.
#    a : float
#        fractional power
#
#    Returns
#    -------
#    data : numpy array
#        The transformed signal.
#
#
#    References
#    ---------
#     .. [1] This algorithm implements `frft.m` from
#        https://nalag.cs.kuleuven.be/research/software/FRFT/
#
#    """
#    ret = np.zeros_like(f, dtype=np.complex)
#    f = f.copy().astype(np.complex)
#    N = len(f)
#    shft = np.fmod(np.arange(N) + np.fix(N / 2), N).astype(int)
#    sN = np.sqrt(N)
#    a = np.remainder(a, 4.0)
#
#    # Special cases
#    if a == 0.0:
#        return f
#    if a == 2.0:
#        return np.flipud(f)
#    if a == 1.0:
#        ret[shft] = np.fft.fft(f[shft]) / sN
#        return ret
#    if a == 3.0:
#        ret[shft] = np.fft.ifft(f[shft]) * sN
#        return ret
#
#    # reduce to interval 0.5 < a < 1.5
#    if a > 2.0:
#        a = a - 2.0
#        f = np.flipud(f)
#    if a > 1.5:
#        a = a - 1
#        f[shft] = np.fft.fft(f[shft]) / sN
#    if a < 0.5:
#        a = a + 1
#        f[shft] = np.fft.ifft(f[shft]) * sN
#
#    # the general case for 0.5 < a < 1.5
#    alpha = a * np.pi / 2
#    tana2 = np.tan(alpha / 2)
#    sina = np.sin(alpha)
#    f = np.hstack((np.zeros(N - 1), sincinterp(f), np.zeros(N - 1))).T
#
#    # chirp premultiplication
#    chrp = np.exp(-1j * np.pi / N * tana2 / 4 *
#                     np.arange(-2 * N + 2, 2 * N - 1).T ** 2)
#    f = chrp * f
#
#    # chirp convolution
#    c = np.pi / N / sina / 4
#    ret = scisig.fftconvolve(
#        np.exp(1j * c * np.arange(-(4 * N - 4), 4 * N - 3).T ** 2),
#        f
#    )
#    ret = ret[4 * N - 4:8 * N - 7] * np.sqrt(c / np.pi)
#
#    # chirp post multiplication
#    ret = chrp * ret
#
#    # normalizing constant
#    ret = np.exp(-1j * (1 - a) * np.pi / 4) * ret[N - 1:-N + 1:2]
#
#    return ret
#
#
#def ifrft(f, a):
#    """
#    Calculate the inverse fast fractional fourier transform.
#
#    Parameters
#    ----------
#    f : np array
#        The signal to be transformed.
#    a : float
#        fractional power
#
#    Returns
#    -------
#    data : np array
#        The transformed signal.
#
#    """
#    return frft(f, -a)
#
#
#def sincinterp(x):
#    N = len(x)
#    y = np.zeros(2 * N - 1, dtype=x.dtype)
#    y[:2 * N:2] = x
#    xint = scisig.fftconvolve(
#        y[:2 * N],
#        np.sinc(np.arange(-(2 * N - 3), (2 * N - 2)).T / 2),
#    )
#    return xint[2 * N - 3: -2 * N + 3]
#    
#    
#    
#
#def fractional3D_forward(fc,a):
#
#
#
#   fc= np.squeeze(fc,axis=1)
#   dim0 = fc.shape[0]
#   dim1 = fc.shape[1]
#   dim2= fc.shape[2]
#   data_matrix = np.ones((dim0,dim1,dim2),np.complex)
#   
#   for ii in range(dim0):
#       for jj in range(dim1):
#         data_matrix[ii,jj,:]= np.fft.fftshift(frft(np.fft.fftshift(fc[ii,jj,:]),a))*np.sqrt(dim2)
#   
#   return data_matrix
#   
#   
#def fractional3D_inverse(fc,a):
#
#
#
#   # fc= np.squeeze(fc,axis=1)
#   dim0 = fc.shape[0]
#   dim1 = fc.shape[1]
#   dim2= fc.shape[2]
#   data_matrix = np.ones((dim0,dim1,dim2),np.complex)
#   
#   for ii in range(dim0):
#       for jj in range(dim1):
#         data_matrix[ii,jj,:]= np.fft.fftshift(ifrft(np.fft.fftshift(fc[ii,jj,:]),a))/np.sqrt(dim2)
#   
#   return data_matrix
#   
#   
#
#X= np.random.rand(2,1,2,2)*3.2+1j
#Y = fractional3D_forward(X,0.4)
#print(Y)
#print('-----')
#print(np.fft.fft(X))
#
#print('********')
#print(fractional3D_inverse(Y,0.4))
#print('******')
#print(np.fft.ifft(Y))
#
#
#
#
##Z= np.array([1+1j, 2+2j, 3 ,4])
##y=np.fft.fftshift(frft(np.fft.fftshift(Z),1.0)*np.sqrt(4))
##print( np.fft.fftshift(frft(np.fft.fftshift(Z),1.0)*np.sqrt(4)))
##
##print(np.fft.fftshift(ifrft(np.fft.fftshift(y),1.0)/np.sqrt(4)))
##print('------------------')
##print(np.fft.fft(Z))
###
#
#
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
#tensor = tf.constant([[1,2,3],[4,5,6],[7,8,9]])
#print("Tensor = ",tensor)
#array = tensor.eval(session=tf.Session())
#print("Array = ",array)





#
#import numpy as np
#
#
#def bizdec(x):
#    k = np.arange(0,np.shape(x)[-1]-1,2)
#    x = x[:,k]
#    return x
#
#
#def bizinter(x):
# 
#    
#    im = 0
#    xmint =0
#    N = np.shape(x)[-1]
#    if  np.sum(np.absolute(np.imag(x)))>0:
#        im=1
#        imx=np.imag(x)
#        x=np.real(x)
#        
#    x2 = np.zeros((np.shape(x)[0],2*N))
#    x2[:,::2] = x
#    xf = np.fft.fft(x2)
#    x_dummy = np.hstack((xf[:,0:int(N/2)],np.zeros((np.shape(x)[0],N)),xf[:,3*int(N/2):2*N]))
#    xint = 2*np.real(np.fft.ifft(x_dummy))
#    
#    if im ==1:
#        x=imx
#        x2 = np.zeros((np.shape(x)[0], 2 * N))
#        x2[:, ::2] = x
#        xf = np.fft.fft(x2)
#        x_dummy = np.hstack((xf[:, 0:int(N / 2)], np.zeros((np.shape(x)[0], N)), xf[:, 3 * int(N / 2):2 * N]))
#        xmint= 2 * np.real(np.fft.ifft(x_dummy))
#        
#    xint = xint+ 1j*xmint
#
#    return xint
#
#def corefrmod2(fc,a):
#    N = np.shape(fc)[-1]
#    deltax = np.sqrt(N)
#    phi = a*np.pi/2
#    alpha = 1/(np.tan(phi))
#    beta = 1/(np.sin(phi))
#    x = np.arange(-np.ceil(N/2),np.fix(N/2))
#    x = np.expand_dims(x,axis=0)/deltax
#    # fc = fc[:,:N]
#    f1 = np.exp(-1j*np.pi*np.tan(phi/2)*x*x)
#    fc = fc*f1
#    del x
#    t = np.arange(-N+1,N)/deltax
#    t = np.expand_dims(t,axis=0)
#    hlptc = np.exp(1j*np.pi*beta*t*t)
#    del t
#    N2 = np.shape(hlptc)[-1]
#    N3 = int(2**(np.ceil(np.log(N2+N-1)/np.log(2))))
#    hlptcz = np.hstack((hlptc,np.zeros((np.shape(hlptc)[0],N3-N2))))
#    fcz = np.hstack((fc,np.zeros((np.shape(fc)[0],N3-N))))
#    Hcfft = np.fft.ifft(np.fft.fft(fcz)*np.fft.fft(hlptcz))
#    del hlptcz
#    del fcz
#    Hc = Hcfft[:,(N-1):(2*N-1)] 
#    del Hcfft
#    del hlptc
#    
#    Aphi = np.exp(-1j*(np.pi*np.sign(np.sin(phi))/4-phi/2))/np.sqrt(np.abs(np.sin(phi)))
#    res = (Aphi*f1*Hc)/deltax
#    del f1
#    del Hc
#    return res
#
#
#def fracF(fc,a):
#
#          
#            batch_matrix= np.squeeze(fc)
#            batch_size = np.shape(batch_matrix)[0]
#            res_matrix = np.zeros((batch_size, np.shape(batch_matrix)[1],np.shape(batch_matrix)[2]),dtype="complex_")
#          
#            for bs in range(batch_size):
#            
#              fc = np.fft.fftshift(batch_matrix[bs,:,:])
#             
#              N = np.shape(fc)[-1]
#              fc = bizinter(fc)
#              fc = np.hstack((np.zeros((np.shape(fc)[0],N)),fc,np.zeros((np.shape(fc)[0],N))))
#              res = fc
#              res = corefrmod2(fc, a)
#              res = res[:,N:(3*N)]
#              res = bizdec(res)
#              res[:,0] = 2*res[:,0]
#             
#              res_matrix[bs,:,:]=np.fft.fftshift(res)
#            return res_matrix
#            
#
#
#
#data= np.array([[[1,2,3,4,5,6],[1,2,3,4,5,6]],[[1,2,3,4,5,6],[1,2,3,4,5,6]]]);
#
#yy = fracF(data,1.0)
#
#print(yy)
            
            
#
#
#import numpy as np
#
#
#
#def bizdec(x):
#    k = np.arange(0,np.shape(x)[-1]-1,2)
#    x = x[:,k]
#    return x
#
#
#def bizinter(x):
#    
#    im = 0
#    xmint =0
#    N = np.shape(x)[-1]
#    if  np.sum(np.absolute(np.imag(x)))>0:
#        im=1
#        imx=np.imag(x)
#        x=np.real(x)
#        
#    x2 = np.zeros((np.shape(x)[0],2*N))
#    x2[:,::2] = x
#    xf = np.fft.fft(x2)
#    x_dummy = np.hstack((xf[:,0:int(N/2)],np.zeros((np.shape(x)[0],N)),xf[:,3*int(N/2):2*N]))
#    xint = 2*np.real(np.fft.ifft(x_dummy))
#    
#    if im ==1:
#        x=imx
#        x2 = np.zeros((np.shape(x)[0], 2 * N))
#        x2[:, ::2] = x
#        xf = np.fft.fft(x2)
#        x_dummy = np.hstack((xf[:, 0:int(N / 2)], np.zeros((np.shape(x)[0], N)), xf[:, 3 * int(N / 2):2 * N]))
#        xmint= 2 * np.real(np.fft.ifft(x_dummy))
#    
#   
#    xint = xint+ 1j*xmint
#
#    return xint
#
#def corefrmod2(fc,a):
#    N = np.shape(fc)[-1]
#    deltax = np.sqrt(N)
#    phi = a*np.pi/2
#    alpha = 1/(np.tan(phi))
#    beta = 1/(np.sin(phi))
#    x = np.arange(-np.ceil(N/2),np.fix(N/2))
#    x = np.expand_dims(x,axis=0)/deltax
#    # fc = fc[:,:N]
#    f1 = np.exp(-1j*np.pi*np.tan(phi/2)*x*x)
#    fc = fc*f1
#    del x
#    t = np.arange(-N+1,N)/deltax
#    t = np.expand_dims(t,axis=0)
#    hlptc = np.exp(1j*np.pi*beta*t*t)
#    del t
#    N2 = np.shape(hlptc)[-1]
#    N3 = int(2**(np.ceil(np.log(N2+N-1)/np.log(2))))
#    hlptcz = np.hstack((hlptc,np.zeros((np.shape(hlptc)[0],N3-N2))))
#    fcz = np.hstack((fc,np.zeros((np.shape(fc)[0],N3-N))))
#    Hcfft = np.fft.ifft(np.fft.fft(fcz)*np.fft.fft(hlptcz))
#    del hlptcz
#    del fcz
#    Hc = Hcfft[:,(N-1):(2*N-1)] 
#    del Hcfft
#    del hlptc
#    
#    Aphi = np.exp(-1j*(np.pi*np.sign(np.sin(phi))/4-phi/2))/np.sqrt(np.abs(np.sin(phi)))
#    res = (Aphi*f1*Hc)/deltax
#    del f1
#    del Hc
#    return res
#
#
#def fracFF(fc,a):
#          
#          N = np.shape(fc)[-1]
#          fc = bizinter(fc)
#          fc = np.hstack((np.zeros((np.shape(fc)[0],N)),fc,np.zeros((np.shape(fc)[0],N))))
#          res = fc
#          res = corefrmod2(fc, a)
#          res = res[:,N:(3*N)]
#          res = bizdec(res)
#          res[:,0] = 2*res[:,0]
#          
#          return res
#
#
#def iterative_fracFF(fc,a):
#
#            batch_matrix= np.squeeze(fc)
#            batch_size = np.shape(batch_matrix)[0]
#            res_matrix = np.zeros((batch_size, np.shape(batch_matrix)[1],np.shape(batch_matrix)[2]),dtype="complex_")
#
#            for bs in range(0,batch_size):
#
#              aa = np.squeeze(batch_matrix[bs,:,:])
#              bb =fracFF(aa,a)
#              res_matrix[bs,:,:]=bb
#
#            return res_matrix
#
#


import numpy as np
import tensorflow as tf



def bizdec(x):
    k = np.arange(0,np.shape(x)[-1]-1,2)
    x = x[:,k]
    return x


def bizinter(x):
    
    im = 0
    xmint =0
    N = np.shape(x)[-1]
    if  np.sum(np.absolute(np.imag(x)))>0:
        im=1
        imx=np.imag(x)
        x=np.real(x)
        
    x2 = np.zeros((np.shape(x)[0],2*N))
    x2[:,::2] = x
    xf = np.fft.fft(x2)
    x_dummy = np.hstack((xf[:,0:int(N/2)],np.zeros((np.shape(x)[0],N)),xf[:,3*int(N/2):2*N]))
    xint = 2*np.real(np.fft.ifft(x_dummy))
    
    if im ==1:
        x=imx
        x2 = np.zeros((np.shape(x)[0], 2 * N))
        x2[:, ::2] = x
        xf = np.fft.fft(x2)
        x_dummy = np.hstack((xf[:, 0:int(N / 2)], np.zeros((np.shape(x)[0], N)), xf[:, 3 * int(N / 2):2 * N]))
        xmint= 2 * np.real(np.fft.ifft(x_dummy))
    
   
    xint = xint+ 1j*xmint

    return xint

def corefrmod2(fc,a):
    N = np.shape(fc)[-1]
    deltax = np.sqrt(N)
    phi = a*np.pi/2
    alpha = 1/(np.tan(phi))
    beta = 1/(np.sin(phi))
    x = np.arange(-np.ceil(N/2),np.fix(N/2))
    x = np.expand_dims(x,axis=0)/deltax
    # fc = fc[:,:N]
    f1 = np.exp(-1j*np.pi*np.tan(phi/2)*x*x)
    fc = fc*f1
    del x
    t = np.arange(-N+1,N)/deltax
    t = np.expand_dims(t,axis=0)
    hlptc = np.exp(1j*np.pi*beta*t*t)
    del t
    N2 = np.shape(hlptc)[-1]
    N3 = int(2**(np.ceil(np.log(N2+N-1)/np.log(2))))
    hlptcz = np.hstack((hlptc,np.zeros((np.shape(hlptc)[0],N3-N2))))
    fcz = np.hstack((fc,np.zeros((np.shape(fc)[0],N3-N))))
    Hcfft = np.fft.ifft(np.fft.fft(fcz)*np.fft.fft(hlptcz))
    del hlptcz
    del fcz
    Hc = Hcfft[:,(N-1):(2*N-1)] 
    del Hcfft
    del hlptc
    
    Aphi = np.exp(-1j*(np.pi*np.sign(np.sin(phi))/4-phi/2))/np.sqrt(np.abs(np.sin(phi)))
    res = (Aphi*f1*Hc)/deltax
    del f1
    del Hc
    return res


def fracFF(fc,a):
          
          N = np.shape(fc)[-1]
          fc = bizinter(fc)
          fc = np.hstack((np.zeros((np.shape(fc)[0],N)),fc,np.zeros((np.shape(fc)[0],N))))
          res = fc
          res = corefrmod2(fc, a)
          res = res[:,N:(3*N)]
          res = bizdec(res)
          res[:,0] = 2*res[:,0]
          
          return res


def iterative_fracFF(fc,a):
         
            
            
            
            with tf.Session() as sess2:
               sess2.run(tf.global_variables_initializer())
               fc=sess2.run(fc)
               type(fc)
            sess2.close()
            
            batch_matrix= np.squeeze(fc)
            batch_size = np.shape(batch_matrix)[0]
            res_matrix = np.zeros((batch_size, np.shape(batch_matrix)[1],np.shape(batch_matrix)[2]),dtype="complex_")

            for bs in range(0,batch_size):

              aa = np.squeeze(batch_matrix[bs,:,:])
              bb =fracFF(aa,a)
              res_matrix[bs,:,:]=bb
     
            return res_matrix

    
    
#data= np.random.randint(1,3,(3,3,10))
#
#xx=np.fft.fftshift(np.fft.fft(np.fft.fftshift(data,axes=2)),axes=2)/np.sqrt(10)
#print(xx)
#
#print('-------------------------')
#
#yy = iterative_fracFF(data,1.0)
#print(yy)

#tf.compat.v1.enable_v2_behavior()
#
#
#aa = tf.random.normal((3,5,6))
#bb=aa
#
#zz= iterative_fracFF(aa,1.0)
#print(zz)
#
#
#print('-------------------------')
##kk=np.fft.fftshift(np.fft.fft(np.fft.fftshift(bb,axes=2)),axes=2)/np.sqrt(6)
#aa=tf.cast(aa,tf.complex64)
#kk=tf.spectral.fft(aa)
#print(kk)

