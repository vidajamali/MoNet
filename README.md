# MoNet ([https://doi.org/10.1073/pnas.2017616118](https://doi.org/10.1073/pnas.2017616118))


## Authors

[Vida Jamali](https://vidajamali.github.io), [Cory Hargus](https://github.com/chargus), Assaf Ben-Moshe, [Amirali Aghazadeh](https://amirmohan.github.io), Hyun Dong Ha, [Kranthi Mandadapu](http://www.cchem.berkeley.edu/kranthi/), and [A. Paul Alivisatos](http://www.cchem.berkeley.edu/pagrp/index.html)

* * * * * *
## Abstract

The motion of nanoparticles near surfaces is of fundamental importance in physics, biology, and chemistry. Liquid cell transmission electron microscopy (LCTEM) is a promising technique for studying motion of nanoparticles with high spatial resolution. Yet, the lack of understanding of how the electron beam of the microscope affects the particle motion has held back advancement in using LCTEM for in situ single nanoparticle and macromolecule tracking at interfaces. Here, we experimentally studied the motion of a model system of gold nanoparticles dispersed in water and moving adjacent to the silicon nitride membrane of a commercial LC in a broad range of electron beam dose rates. We find that the nanoparticles exhibit anomalous diffusive behavior modulated by the electron beam dose rate. We characterized the anomalous diffusion of nanoparticles in LCTEM using a convolutional deep neural-network model and canonical statistical tests. The results demonstrate that the nanoparticle motion is governed by fractional Brownian motion at low dose rates, resembling diffusion in a viscoelastic medium, and continuous-time random walk at high dose rates, resembling diffusion on an energy landscape with pinning sites. Both behaviors can be explained by the presence of silanol molecular species on the surface of the silicon nitride membrane and the ionic species in solution formed by radiolysis of water in presence of the electron beam.

* * * * * *
### MoNet Description
MoNet is a trainable convolutional neural network that classifies the behavior of a trajectory from single particle tracking experiments based on three classes of diffusion (Brownian, subdiffusive Fractional Brownian Motion (FBM), and subdiffusive continuous time random walk (CTRW)) and extracts the α exponent for FBM as well as CTRW classes. This is the code for the paper [Anomalous nanoparticle surface diffusion in LCTEM is revealed by deep learning-assisted analysis](https://doi.org/10.1073/pnas.2017616118). The architecture of MoNet is adapted from [Granik et al.](https://github.com/AnomDiffDB/DB) and is modified based on the [p-variation method](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.180602) to capture multiresolution correlations along the trajectory.

### MoNet Architecture
<p align="center">
<img src="./images/MoNet-arc.png" width="600" /> </p>

The architecture of MoNet comprises of 4 layers where the first layer consists of 6 convolutional sublayers (f<sub>11</sub>, f<sub>12</sub>, f<sub>13</sub>, f<sub>14</sub>, f<sub>15</sub>, f<sub>16</sub>) that are applied on the input data in parallel. The first 5 convolutuonal sublayers are three layers deep with relu activation units (relu(·) = max(·,0) for rectification of the feature map), batch normalization (normalizing the responses across features map), and max pooling (finding the maximum over a local neighborhood). The number of filters applied in all of these sublayers are set to 32. After training, each of these filters capture a certain distinct pattern along the trajectory (e.g., descending, ascending patterns). The diversity among the filters are typically ensured via random initialization of the filters and regularization techniques such as batch normalization and drop out. The filter sizes are k = 3, 4, 2, 10, and 20 respectively for the five convolutional sublayers to capture the local dynamics of trajectories in several spacial resolutions. The convolutional sublayers also differ in their dilation factor (i.e., the number of steps that filters skip). Following p-variation we chose dilation factors that span the trajectory via steps of size 2<sup>n</sup>. The last convolutional sublayer, f<sub>16</sub> augments the model using large filter sizes of length 20 without any dilation. The output of the convolutional sublayers are fed into two fully connected layers of size 512 and 128 (f<sub>2</sub> and f<sub>3</sub>, respectively). The final layer of MoNet (f<sub>4</sub>) is set based on the prediction task. For the anomalous classification task, the last layer is a dense layer of size 3 (corresponding to the three classes of diffusion) with a Softmax activation. For the regression task of finding the α exponent, a dense layer of size 1 with a Sigmoid activation is used in the last layer to capture the output.

### Installation

- install [anaconda](https://docs.anaconda.com/anaconda/install/)
- conda env create -f MoNet.yml
- conda activate MoNet
- conda install -c anaconda keras
- `pip install matplotlib`
- `pip install stochastic`
- `pip install seaborn`

### MoNet Training

For the task of classification, the network is trained on 10,000 simulated trajectories from three classes of diffusion (FBM, Brownian, and CTRW) with randomly chosen parameters. For the task of α extraction the network is trained on 3,000 simulated trajectories (from FBM or CTRW, depending on the code used). We used the same architecture universally regardless of the task (regression/classification). In the example used here the trajectories are 300 frames long; however, the code can be changed to work for any other trajectory length N. A new model needs to be trained for any new trajectory length, which requires changing the testing file as well. 

### Testing using MoNet

For testing data from experiments, MoNet received data in `.mat` format via `classification_on_file(file = './data/AuNRs_300.mat')`. The input data must have the structure of `[x,y,k]`, where k is the index for the trajectory that starts from 1 and goes up to the total number of trajectories that needs to be tested. The example data set included in the ./data folder (`AuNRs_300.mat`) includes all the 300-frame long trajectories collected from our liquid cell TEM experiments for all nanoparticles, indexed from 1 to 459. Please refer to the manuscript for the details about the data collection.

For testing simulated data change the classification testing file to `classification_on_sim()`. 

### Citation
If you are using this code, please reference our paper:
```
  @article{jamali2020anomalous,
    title={Anomalous Nanoparticle Surface Diffusion in Liquid Cell TEM is Revealed by Deep Learning-Assisted Analysis},
    author={Jamali, Vida and Hargus, Cory and Ben Moshe, Assaf and Aghazadeh, Amirali and Ha, Hyun Dong and Mandadapu, Kranthi K and Alivisatos, Paul},
    volume = {118},
	  number = {10},
	  elocation-id = {e2017616118},
	  year = {2021},
	  doi = {10.1073/pnas.2017616118},
    publisher={PNAS}
    }
```    

### Funding
This research was supported by the NSF, Division of Chemical, Bioengineering, Environmental, and Transport Systems under Award 2039624. C.H. is supported by the NSF Graduate Research Fellowship Program under Grant DGE-1752814. H.D.H. acknowledges the Samsung Scholarship for a graduate fellowship. K.K.M. is partially supported by Director, Office of Science, Office of Basic Energy Sciences, of the US Department of Energy under Contract DEAC02-05-CH11231 FWP CHPHYS02. A.B.-M. is supported by the US Department of Energy, Office of Science, Office of Basic Energy Sciences, Materials Sciences and Engineering Division, under Contract DE-AC02-05CH11231 within the Characterization of Functional Nanomachines Program under Award KC1203.
