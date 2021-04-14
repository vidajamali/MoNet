"""
MoNet_classification_testing.py
This script contains functions for activating and testing of the classification
net. 

"""  
 
import numpy as np
from keras.models import load_model
from utils import generate_sim
import scipy.io
import pickle






def classification_on_sim():
    dx,label=generate_sim(batchsize=20,steps=300,T=15,sigma=0.1)
    ### change here to load a different network model
    N=np.shape(dx)[0]
    net_file = './Models/FINALmodel_300.h5'
    model = load_model(net_file)     
    for j in range(N):
        dummy = np.zeros((1,299,1))
        dummy[0,:,:] = dx[j,:,:]
        prob_pred = model.predict(dummy) # get the results for 1D 
        probmean = np.mean(prob_pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        prediction = np.argmax(probmean,axis=0) # translate to classification
        print('prob_pred {}'.format(prob_pred))
        print('prediction {}'.format(prediction))
        print('ground truth',label[j])
        print('--')

    
    return


"""
Function classification_on_file is used to classify trajectories loaded from a
.mat file

The function assumes the input comes in the form x,y,z,...,N where N is the 
trajectory index number, starting from one.

Input: 
    file - string containing the file name, ending with .mat
    
Outputs:
    prediction - Classification to diffusion model type where 0=FBM; 1=Brownian; 2=CTRW
    prob_full - probability values associated with each diffusion class for any input 
    trajectory
"""
    
    
    
def classification_on_file(file):
    ### change here to load a different network model
    net_file = './Models/FINALmodel_300.h5'
    model = load_model(net_file)

    # load mat file and extract trajectory data
    
    f = scipy.io.loadmat(file)
    for k in f.keys():
        if k[0]=='_':
            continue
        varName = k
        data = f[varName]
        NAxes = (np.shape(data)[1]-1)
    
    numTraj = len(np.unique(data[:,NAxes]))
    prediction = np.zeros([numTraj,1])
    prob_full = np.zeros([numTraj,3])
    flag = np.zeros([numTraj,1])
    for i in np.arange(1,numTraj+1):
        prob_pred = np.zeros([NAxes,3])
        for j in range(NAxes):
            x = data[np.argwhere(data[:,NAxes]==i),j]
            x = x-np.mean(x)
            if len(x)>=300: # classification network is trained on 300 step trajectories
                flag[i-1] = 1 # mark trajectories that are being analyzed
                dx = np.diff(x,axis=0)
                variance= np.std(dx)
                dx =dx/variance
                dx = np.reshape(dx[:299],[1,299,1]) #change this number based on step size
            prob_pred[j,:] = model.predict(dx) # get the results for 1D 
        probmean = np.mean(prob_pred,axis=0) # calculate mean prediction of N-dimensional trajectory 
        prediction[i-1,0] = np.argmax(probmean,axis=0) # translate to classification
        prob_full[i-1,:] = probmean
    prediction = prediction[np.where(flag==1)]
    
    return prediction,prob_full


# prediction,prob_full = classification_on_sim()
# print(prediction)
prediction,prob_full = classification_on_file(file = './data/AuNRs_300.mat')
print(prediction.shape)
print(prediction)
pickle.dump( prediction, open( "./results/prediction-AuNRs.p", "wb" ) )
pickle.dump( prob_full, open( "./results/probfull-AuNRs.p", "wb" ) )