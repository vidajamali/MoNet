"""
utils.py
This script contains functions for generating diffusion simulations, 
data generators needed for the network training/testing, and other necessary 
functions.
Original version by Granik et al is accessible at:  https://github.com/AnomDiffDB/DB
Updated version of this function has bugs fixed on the standard deviation of the data 
generated from different classes of diffusions, heavy-tailed distribution of waiting times
 in a CTRW class, and new functions are added to simulate hybrid trajectories 
"""


import numpy as np
from scipy import stats,fftpack
from keras.utils import to_categorical
from stochastic import diffusion 
import scipy.io


"""
Function autocorr calculates the autocorrelation of a given input vector x

Input: 
    x - 1D vector 
    
Outputs:
    autocorr(x)    
"""

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[np.int(result.size/2):]


"""
Function OrnsteinUng generates a single realization of the Ornstein–Uhlenbeck 
noise process
see https://stochastic.readthedocs.io/en/latest/diffusion.html#stochastic.diffusion.OrnsteinUhlenbeckProcess
for more details.

Input: 
    n - number of points to generate
    T - End time
    speed - speed of reversion 
    mean - mean of the process
    vol - volatility coefficient of the process
    
Outputs:
    x - Ornstein Uhlenbeck process realization
"""

def OrnsteinUng(n=1000,T=50,speed=0,mean=0,vol=0):
    OU = diffusion.OrnsteinUhlenbeckProcess(speed=speed,mean=mean,vol=vol,t=T)
    x = OU.sample(n=n)
    
    return x

#%% 
'''
function fbm_diffusion generates FBM diffusion trajectory (x,y,t)
realization is based on the Circulant Embedding method presented in:
Schmidt, V., 2014. Stochastic geometry, spatial statistics and random fields. Springer.

Input: 
    n - number of points to generate
    H - Hurst exponent
    T - end time
    
Outputs:
    x - x axis coordinates
    y - y axis coordinates
    t - time points
        
'''
def fbm_diffusion(n=1000,H=1,T=15):

    # first row of circulant matrix
    r = np.zeros(n+1)
    r[0] = 1
    idx = np.arange(1,n+1,1)
    r[idx] = 0.5*((idx+1)**(2*H) - 2*idx**(2*H) + (idx-1)**(2*H))
    r = np.concatenate((r,r[np.arange(len(r)-2,0,-1)]))
    
    # get eigenvalues through fourier transform
    lamda = np.real(fftpack.fft(r))/(2*n)
    
    # get trajectory using fft: dimensions assumed uncoupled
    x = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    x = n**(-H)*np.cumsum(np.real(x[:n])) # rescale
    x = ((T**H)*x)# resulting traj. in x
    y = fftpack.fft(np.sqrt(lamda)*(np.random.normal(size=(2*n)) + 1j*np.random.normal(size=(2*n))))
    y = n**(-H)*np.cumsum(np.real(y[:n])) # rescale
    y = ((T**H)*y) # resulting traj. in y

    t = np.arange(0,n+1,1)/n
    t = t*T # scale for final time T
    

    return x,y,t

'''
CTRW diffusion - generate CTRW trajectory (x,y,t)
function based on mittag-leffler distribution for waiting times and 
alpha-levy distribution for spatial lengths.
for more information see: 
Fulger, D., Scalas, E. and Germano, G., 2008. 
Monte Carlo simulation of uncoupled continuous-time random walks yielding a 
stochastic solution of the space-time fractional diffusion equation. 
Physical Review E, 77(2), p.021122.

Inputs: 
    n - number of points to generate
    alpha - exponent of the waiting time distribution function 
    gamma  - scale parameter for the mittag-leffler and alpha stable distributions.
    T - End time
'''
# Generate mittag-leffler random numbers
def mittag_leffler_rand(beta=0.5, n=1000, gamma=1):
    t = -np.log(np.random.uniform(size=[n, 1]))
    u = np.random.uniform(size=[n, 1])
    w = np.sin(beta * np.pi) / np.tan(beta * np.pi * u) - np.cos(beta * np.pi)
    t = t * w**(1. / beta)
    t = gamma * t

    return t


# Generate symmetric alpha-levi random numbers
def symmetric_alpha_levy(alpha=0.5, n=1000, gamma=1):
    u = np.random.uniform(size=[n, 1])
    v = np.random.uniform(size=[n, 1])

    phi = np.pi * (v - 0.5)
    w = np.sin(alpha * phi) / np.cos(phi)
    z = -1 * np.log(u) * np.cos(phi)
    z = z / np.cos((1 - alpha) * phi)
    x = gamma * w * z**(1 - (1 / alpha))

    return x


# needed for CTRW
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


# Generate CTRW diffusion trajectory
def CTRW(n=1000, alpha=1, gamma=1, T=40):
    '''
    CTRW diffusion - generate CTRW trajectory (x,y,t)
    function based on mittag-leffler distribution for waiting times and
    alpha-levy distribution for spatial lengths.
    for more information see:
    Fulger, D., Scalas, E. and Germano, G., 2008.
    Monte Carlo simulation of uncoupled continuous-time random walks yielding a
    stochastic solution of the space-time fractional diffusion equation.
    Physical Review E, 77(2), p.021122.

    https://en.wikipedia.org/wiki/Lévy_distribution
    https://en.wikipedia.org/wiki/Mittag-Leffler_distribution

    Inputs:
        n - number of points to generate
        alpha - exponent of the waiting time distribution function
        gamma  - scale parameter for the mittag-leffler and alpha stable
                 distributions.
        T - End time
    '''
    jumpsX = mittag_leffler_rand(alpha, n, gamma)

    rawTimeX = np.cumsum(jumpsX)
    tX = rawTimeX * (T) / np.max(rawTimeX)
    tX = np.reshape(tX, [len(tX), 1])

    x = symmetric_alpha_levy(alpha=2, n=n, gamma=gamma**(alpha / 2))
    x = np.cumsum(x)
    x = np.reshape(x, [len(x), 1])

    y = symmetric_alpha_levy(alpha=2, n=n, gamma=gamma**(alpha / 2))
    y = np.cumsum(y)
    y = np.reshape(y, [len(y), 1])

    tOut = np.arange(0, n, 1) * T / n
    xOut = np.zeros([n, 1])
    yOut = np.zeros([n, 1])
    for i in range(n):
        xOut[i, 0] = x[find_nearest(tX, tOut[i]), 0]
        yOut[i, 0] = y[find_nearest(tX, tOut[i]), 0]
    return xOut.T[0], yOut.T[0], tOut


'''
Brownian - generate Brownian motion trajectory (x,y)

Inputs: 
    N - number of points to generate
    T - End time 
    delta - Diffusion coefficient

Outputs:
    out1 - x axis values for each point of the trajectory
    out2 - y axis values for each point of the trajectory
'''

def Sub_brownian(x0, n, dt, delta, out=None):
    x0 = np.asarray(x0)
    # generate a sample of n numbers from a normal distribution.
    r = stats.norm.rvs(size=x0.shape + (n,), scale=delta*np.sqrt(dt))

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)
    # Compute Brownian motion by forming the cumulative sum of random samples. 
    np.cumsum(r, axis=-1, out=out)
    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out

def Brownian(N=1000,T=50,delta=1):
    x = np.empty((2,N+1))
    x[:, 0] = 0.0
    
    Sub_brownian(x[:,0], N, T/N, delta, out=x[:,1:])
    
    out1 = x[0]
    out2 = x[1]
    
    return out1,out2


#%%
'''
Generator functions for neural network training per Keras specifications
input for all functions is as follows:
    
input: 
   - batch size
   - steps: total number of steps in trajectory (list) 
   - T: final time (list)
   - sigma: Standard deviation of localization noise (std of a fixed cell/bead)
'''

def generate(batchsize=32,steps=1000,T=15,sigma=0.1):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps-1,1])
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            # randomly select diffusion model to simulate for this iteration
            label[i,0] = np.random.choice([0,1,2])
            if label[i,0] == 0: 
                H = np.random.uniform(low=0.1,high=0.48) #subdiffusive
                x,y,t = fbm_diffusion(n=steps,H=H,T=T1)                  
            elif label[i,0] == 1:
                x,y = Brownian(N=steps,T=T1,delta=1) 
            else:
                alpha=np.random.uniform(low=0.1,high=0.99)
                x,y,t = CTRW(n=steps,alpha=alpha,T=T1)
            noise = np.sqrt(sigma)*np.random.randn(steps-1)
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)
            x_n = x1[0,:steps]
            dx = np.diff(x_n)
            # Generate OU noise to add to the data
#             nx = OrnsteinUng(n=steps-2,T=T1,speed=1,mean=0,vol=1)
#             dx = dx+sigma*nx
            if np.std(x) < 0.000001:
                dx = dx
            else:
                dx = dx/np.std(dx)
                dx = dx+noise
                out[i,:,0] = dx
       
        label = to_categorical(label,num_classes=3)
        yield out,label
        

'''
Generator FBM and CTRW trajectories for neural network testing to track the performance of
neural network on simulated data from both of these classes
input for all functions is as follows:
    
input: 
   - batch size
   - steps: total number of steps in trajectory (list) 
   - T: final time (list)
   - sigma: Standard deviation of localization noise (std of a fixed cell/bead)
'''

# Randomly generate trajectories of different diffusion models for training of the 
# classification network
    
def generate_sim(batchsize=32,steps=1000,T=15,sigma=0.1):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps-1,1])
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            # randomly select diffusion model to simulate for this iteration
            label[i,0] = np.random.choice([0,1,2])
            if label[i,0] == 0: 
                H = np.random.uniform(low=0.09,high=0.45) #subdiffusive
                x,y,t = fbm_diffusion(n=steps,H=H,T=T1)
            elif label[i,0] == 1:
                x,y = Brownian(N=steps,T=T1,delta=1) 
            else:
                x,y,t = CTRW(n=steps,alpha=np.random.uniform(low=0.2,high=0.9),T=T1)
            noise = np.sqrt(sigma)*np.random.randn(1,steps)
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)     
            x_n = x1[0,:steps]
            dx = np.diff(x_n)

            if np.std(x) < 0.000001:
                dx = dx
            else:
                dx = dx/np.std(dx)
            out[i,:,0] = dx
       
        return out,label
    
        
# generate FBM trajectories with different Hurst exponent values 
# for training of the Hurst-regression network
        
def generate_fbm(batchsize=32,steps=1000,T=[1],sigma=0.1):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps-1,1])
        
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            H = np.random.uniform(low=0.1,high=0.48)
            label[i,0] = H
            x,y,t = fbm_diffusion(n=steps,H=H,T=T1)
            
            n = np.sqrt(sigma)*np.random.randn(steps,)
            x_n = x[:steps,]+n
            dx = np.diff(x_n,axis=0)
            
            out[i,:,0] = autocorr((dx-np.mean(dx))/(np.std(dx)))

        
        yield out,label
        
     
'''
Generate CTRW for CTRW single for finding alpha value
'''  
def generate_CTRW(batchsize=32,steps=1000,T=15,sigma=0.1):
    while True:
        # randomly choose a set of trajectory-length and final-time. This is intended
        # to increase variability in simuation conditions.
        T1 = np.random.choice(T,size=1).item()
        out = np.zeros([batchsize,steps-1,1])
        label = np.zeros([batchsize,1])
        for i in range(batchsize):
            alpha=np.random.uniform(low=0.1,high=0.99)
            label[i,0] = alpha
            x,y,t = CTRW(n=steps,alpha=alpha,T=T1)
            noise = np.sqrt(sigma)*np.random.randn(steps-1)
            x1 = np.reshape(x,[1,len(x)])
            x1 = x1-np.mean(x1)
            x_n = x1[0,:steps]
            dx = np.diff(x_n)
            if np.std(x) < 0.000001:
                dx = dx
            else:
                dx = dx/np.std(dx)
                dx = dx+noise
            out[i,:,0] = dx
            

        yield out, label       



    """Generate a hybrid trajectory by superposing.

    This function takes as input a CTRW trajectory, an FBM
    trajectory and a weight, and returns a weighted sum of
    the two after normalizing by their RMSD.


    Parameters
    ----------

    xctrw, yctrw: 1D numpy array
        Arrays of the CTRW trajectory from `CTRW` function.

    xfbm, yfbm: 1D numpy array
        Arrays of the FBM trajectory from `fbm_diffusion` function.

    weight_ctrw: float
        A weight between 0 and 1 to apply to the CTRW trajectory. If
        `weight`=0, the trajectory is purely FBM, if `weight`=1, it is
        purely CTRW.

    scale : float
        The input CTRW and FBM trajectories are scaled by their total
        RMSD, to ensure they are comparable. The resulting hybrid trajectory
        can then optionally be scaled by `scale`.

    Returns
    -------
    xhybrid, yhybrid: 1D numpy array
        Arrays of the x and y positions of the hybrid trajectory.

    """
def hybrid_superpose(xctrw, yctrw, xfbm, yfbm, weight=0.5, scale=1):    
    assert(len(xctrw) == len(xfbm))
    xhybrid = weight*xctrw/np.std(xctrw) + (1.-weight)*xfbm/np.std(xfbm)
    yhybrid = weight*yctrw/np.std(yctrw) + (1.-weight)*yfbm/np.std(yfbm)
    return xhybrid, yhybrid



    """Generate a hybrid trajectory by randomly interleaving.

    This function takes as input a CTRW trajectory, an FBM
    trajectory and a weight, and returns a randomly interleaved
    hybrid of the two after normalizing each by their RMSD.


    Parameters
    ----------

    xctrw, yctrw: 1D numpy array
        Arrays of the CTRW trajectory from `CTRW` function.

    xfbm, yfbm: 1D numpy array
        Arrays of the FBM trajectory from `fbm_diffusion` function.

    window : float
        The length (in timesteps, i.e. number of frames) of each
        interval of FBM or CTRW.

    Returns
    -------
    xhybrid, yhybrid: 1D numpy array
        Arrays of the x and y positions of the hybrid trajectory.

    """
def hybrid_interleave(xctrw, yctrw, xfbm, yfbm, window=300):
    
    assert(len(xctrw) == len(xfbm))
    M = window  # rename for convenience
    N = len(xctrw) // M  # Number of windows
    xhybrid = np.empty(N*M)
    yhybrid = np.empty(N*M)
    xcurr=0
    ycurr=0
    rand = np.random.binomial(1,.5, N).astype(bool)
    sel = np.repeat(rand,M)
#     ctrw_frames = np.where(sel)
#     fbm_frames = np.where(~sel)
    for i in range(N):
        if rand[i]:
            xhybrid[i*M:(i+1)*M] = xctrw[i*M:(i+1)*M] / np.std(xctrw)
            yhybrid[i*M:(i+1)*M] = yctrw[i*M:(i+1)*M] / np.std(yctrw)
        else:
            xhybrid[i*M:(i+1)*M] = xfbm[i*M:(i+1)*M] / np.std(xfbm)
            yhybrid[i*M:(i+1)*M] = yfbm[i*M:(i+1)*M] / np.std(yfbm)
        xhybrid[i*M:(i+1)*M] -= (xhybrid[i*M] - xcurr)
        yhybrid[i*M:(i+1)*M] -= (yhybrid[i*M] - ycurr)
        xcurr =  xhybrid[(i+1)*M-1]
        ycurr =  yhybrid[(i+1)*M-1]
    return xhybrid, yhybrid, sel