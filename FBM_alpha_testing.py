"""
FBM_single_functions.py
This script contains functions for activating and testing of Fractional Brownian Motion
single-trajectory networks trained to estimate the Hurst exponent. 
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from utils import fbm_diffusion
from utils import autocorr
import scipy.optimize
import scipy.io
import seaborn as sns
import pickle


"""
Function predict_1D is used to run a version of the network on a single trajectory.
The function assumes the input is a column vector

Input: 
    x - 1D column vector containing localization data
    stepsActual - number of steps to analyze in each trajectory. This number 
                  determnines what network will run on the data.
                  Select from options: [10,25,60,100,200,300,500,700,1000]
    reg_model - loaded model of the network (comes as input to avoid
                reloading the same model for each trajectory)
                  
Outputs:
    pX - Network prediction for the Hurst exponenet of the 1D trajectory.           
"""

def predict_1D(x,stepsActual,reg_model):
    
    if len(x)<stepsActual:
        return 0
    
    dx = (np.diff(x[:stepsActual],axis=0)[:,0])
    dx = autocorr((dx-np.mean(dx))/(np.std(dx)+1e-10))
    dx = np.reshape(dx,[1,np.size(dx),1]) 
    pX = reg_model.predict(dx)
    
    return pX
    
        
 
"""
Function net_on_file is used to run a version of the network on a .mat file
containing one or more single particle trajectories.

The function assumes the input comes in the form x,y,z,...,N where N is the 
trajectory serial number, starting from one.

Input: 
    file - string containing the file name, ending with .mat
                  
Outputs:
    prediction - A vector with lenght as number of trajectories containing 
                 network predictions (average of N-dimensional predictions)
    NDpred - A matrix with dimensions [#trajetories,#dimensions] containing
             all predictions done by the network (N-dimensions for each trajectory)
"""

def FBM_net_on_file(file,stepsActual):
    
    # laod trained keras model
    ### change here to load a different network model
    net_file = './Models/300-H-estimate.h5'
    reg_model = load_model(net_file)
    ###
    
    # load mat file and extract trajectory data
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
        numTraj = len(np.unique(data[:,NAxes]))

    # allocate variables to hold temporary data and prediction results
    prediction = np.zeros([numTraj,1])
    NDpred = np.zeros([numTraj,(np.shape(data)[1]-1)])    
    
    # iterate over trajectories and analyze data
    
    for i in np.arange(0,numTraj):
        for j in range((np.shape(data)[1]-1)):
            x = data[np.argwhere(data[:,NAxes]==i+1),j]
        
            pX = predict_1D(x,stepsActual,reg_model)
        
            NDpred[i,j] = pX
        
    NDpred = NDpred[np.where(NDpred>0)]
    NDpred = np.reshape(NDpred,[int(np.size(NDpred)/NAxes),NAxes])
    
    prediction = np.mean(NDpred,axis=1)
    
    return prediction, NDpred



prediction, NDpred = FBM_net_on_file(file = './data/AuNRs_300.mat',stepsActual=300)
print(prediction.shape)
print(prediction)
pickle.dump( prediction, open( "./results/predH-FBM-AuNRs-300.p", "wb" ) )
pickle.dump( NDpred, open( "./results/NDpred-FBM-AuNRs-300.p", "wb" ) )