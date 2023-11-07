# -*- coding: utf-8 -*-
"""

@author: Tingting Zhu
@contact: tingting.zhu@eng.ox.ac.uk
@reference: 
GPy Library https://github.com/SheffieldML/GPy
GPy Document https://gpy.readthedocs.io/en/deploy/

"""

#%% Fit a univariate Gaussian process to the atmospheric CO2 observations from 
# the Mauna Loa Observatory in Hawaii 

#%% import libraries
import numpy as np
from matplotlib import pyplot as plt
import GPy

def plot_gp(X, m, C, training_points=None):
    """ Plotting utility to plot a GP fit with 95% confidence interval """
    # Plot 95% confidence interval 
    plt.fill_between(X[:,0],
                     m[:,0] - 1.96*np.sqrt(np.diag(C)),
                     m[:,0] + 1.96*np.sqrt(np.diag(C)),
                     alpha=0.5)
    # Plot GP mean and initial training points
    plt.plot(X, m, "-")
    plt.legend(labels=["GP fit"])
    plt.xlabel("x"), plt.ylabel("f")
    
    # Plot training points if included
    if training_points is not None:
        X_, Y_ = training_points
        plt.plot(X_, Y_, "kx", mew=2)
        plt.legend(labels=["GP fit", "sample points"])
        
#%% Description of the tutorial

# Let's consider a real world example using atmospheric CO2 observations from 
# the Mauna Loa Observatory, Hawaii. 

#%% import data from GPy example
GPy.util.datasets.authorize_download = lambda x: True # This gives GPy permission to download the dataset

# Download the data, cache it locally and pass to variable `data`
data = GPy.util.datasets.mauna_loa(refresh_data=False)

print("\nData keys:")
print(data.keys())

print("\nCitation:")
print(data['citation'])

print("\nInfo:")
print(data['info'])

#%% Extraction of train and test data

# Training data (X = input, Y = observation)
X, Y = data['X'], data['Y']
# Test data (Xtest = input, Ytest = observations)
Xtest, Ytest = data['Xtest'], data['Ytest']
Xnew = np.vstack([X, Xtest])

# Plot the training data in blue and the test data in red
plt.figure(figsize=(20, 10))
plt.plot(X, Y, "b.", Xtest, Ytest, "r.")
plt.legend(labels=["training data", "test data"])
plt.xlabel("year"), plt.ylabel("CO$_2$ (PPM)"), plt.title("Monthly mean CO$_2$ at the Mauna Loa Observatory, Hawaii");


#%% Define a kernel or a combinations of kernels

# hint: you will need a combination of kernels to model the overall non-linearity,
# offset, upwards linear trend, short-term periodicity, and long-term amplitude modulation.

# an example of RBF kernel:
kern_RBF = GPy.kern.RBF(1, variance=1., lengthscale=2., name="RBF")

#%% Construct a GP model for regression fitting with the designed kernel


#%% Optimise the GP model


#%% Predict the values for Xnew


#%% Plot the GP fit mean and covariance using plot_gp for Xnew


