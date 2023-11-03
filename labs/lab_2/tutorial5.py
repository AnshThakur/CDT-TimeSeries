# -*- coding: utf-8 -*-
"""

@author: Tingting Zhu
@contact: tingting.zhu@eng.ox.ac.uk
@reference: 
GPy Library https://github.com/SheffieldML/GPy
GPy Document https://gpy.readthedocs.io/en/deploy/

"""

#%% Multi-output/Multi-task Gaussian process

#%% import libraries

import GPy
import numpy as np
from matplotlib import pyplot as plt


#%% define a plotting function
def plot_2outputs(m,xlim,ylim):
    fig = plt.figure(figsize=(12,8))
    # Observation 1
    ax1 = fig.add_subplot(221)
    ax1.set_xlim(xlim)
    ax1.set_title('Obervation 1')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,0)],which_data_rows=slice(0,50),ax=ax1)
    ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
    # Observation 2
    ax2 = fig.add_subplot(222)
    ax2.set_xlim(xlim)
    ax2.set_title('Observation 2')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,1)],which_data_rows=slice(50,100),ax=ax2)
    ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)  
    # Observation 3
    ax3 = fig.add_subplot(223)
    ax3.set_xlim(xlim)
    ax3.set_title('Observation 3')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,2)],which_data_rows=slice(100,150),ax=ax3)
    ax3.plot(Xt3[:,:1],Yt3,'rx',mew=1.5)  
    # Observation 4
    ax4 = fig.add_subplot(224)
    ax4.set_xlim(xlim)
    ax4.set_title('Observation 4')
    m.plot(plot_limits=xlim,fixed_inputs=[(1,3)],which_data_rows=slice(150,200),ax=ax4)
    ax4.plot(Xt4[:,:1],Yt4,'rx',mew=1.5)  


  

#%%This functions generate data corresponding to four observations
f_output1 = lambda x: 4. * np.sin(x/5.) - .4*x + np.random.rand(x.size)[:,None] * 2.
f_output2 = lambda x: 6. * np.sin(x/5.) + .2*x + np.random.rand(x.size)[:,None] * 8.
f_output3 = lambda x: 5. * np.sin(x/5.) - .5*x + np.random.rand(x.size)[:,None] * 5.
f_output4 = lambda x: 10. * np.sin(x/5.) + 1.*x + np.random.rand(x.size)[:,None] * 10.


#{X,Y} training set for each observation
np.random.seed(1000) 
X1 = np.random.rand(50)[:,None]; X1=X1*75
X2 = np.random.rand(50)[:,None]; X2=X2*70 + 30
X3 = np.random.rand(50)[:,None]; X3=X3*90 + 30
X4 = np.random.rand(50)[:,None]; X4=X4*80 + 20

Y1 = f_output1(X1)
Y2 = f_output2(X2)
Y3 = f_output3(X3)
Y4 = f_output4(X4)

#{X,Y} test set for each observation
Xt1 = np.random.rand(50)[:,None]*100
Xt2 = np.random.rand(50)[:,None]*100
Xt3 = np.random.rand(50)[:,None]*100
Xt4 = np.random.rand(50)[:,None]*100

Yt1 = f_output1(Xt1)
Yt2 = f_output2(Xt2)  
Yt3 = f_output3(Xt3) 
Yt4 = f_output4(Xt4)    

# plot four observations
xlim = (0,100); ylim = (0,50)
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(221)
ax1.set_xlim(xlim)
ax1.set_title('Observation 1')
ax1.plot(X1[:,:1],Y1,'kx',mew=1.5,label='Train set')
ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5,label='Test set')
ax1.legend()

ax2 = fig.add_subplot(222)
ax2.set_xlim(xlim)
ax2.set_title('Observation 2')
ax2.plot(X2[:,:1],Y2,'kx',mew=1.5,label='Train set')
ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5,label='Test set')
ax2.legend() 

ax3 = fig.add_subplot(223)
ax3.set_xlim(xlim)
ax3.set_title('Observation 3')
ax3.plot(X3[:,:1],Y3,'kx',mew=1.5,label='Train set')
ax3.plot(Xt3[:,:1],Yt3,'rx',mew=1.5,label='Test set')
ax3.legend()  

ax4 = fig.add_subplot(224)
ax4.set_xlim(xlim)
ax4.set_title('Observation 4')
ax4.plot(X4[:,:1],Y4,'kx',mew=1.5,label='Train set')
ax4.plot(Xt4[:,:1],Yt4,'rx',mew=1.5,label='Test set')
ax4.legend()   

#%% GP fitting without coregionalization

#%% Construct a GP model for regression fitting with the designed kernel


#%% Optimise the GP model
#m1 = 
#m2 = 
#m3 = 
#m4 = 

#%% Plot results of the GP fit for each observation
fig = plt.figure(figsize=(20,10))
# Observation 1
ax1 = fig.add_subplot(221)
m1.plot(plot_limits=xlim,ax=ax1)
ax1.plot(Xt1[:,:1],Yt1,'rx',mew=1.5)
ax1.set_title('Obervation 1')
# Observation 2
ax2 = fig.add_subplot(222)
m2.plot(plot_limits=xlim,ax=ax2)
ax2.plot(Xt2[:,:1],Yt2,'rx',mew=1.5)
ax2.set_title('Obervation 2')
# Observation 3
ax3 = fig.add_subplot(223)
m3.plot(plot_limits=xlim,ax=ax3)
ax3.plot(Xt3[:,:1],Yt3,'rx',mew=1.5)
ax3.set_title('Obervation 3')
# Observation 4
ax4 = fig.add_subplot(224)
m4.plot(plot_limits=xlim,ax=ax4)
ax4.plot(Xt4[:,:1],Yt4,'rx',mew=1.5)
ax4.set_title('Obervation 4')
    
#%% Using the GPy's Coregionalized Regression Modelto model the obervations

#%% Define kernel K and estimate B for corgionalization
#K = 
#B =  

# Print components of B



#%% Define an ICM kernel that deals with multiple obervations

#icm = GPy.util.multioutput.ICM

#%% Construct a GP model for coregionalized regression fitting with the ICM kernel
#an appropiate kernel for our model, its use is straightforward. 
#In the next example we will use a Matern-3/2 kernel as K.

# m = 

# plot results using plot_2outputs
fig = plt.figure(figsize=(20,10))
plot_2outputs(m,xlim=(0,100),ylim=(-50,130))





