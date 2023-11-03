# -*- coding: utf-8 -*-
"""

@author: Tingting Zhu
@contact: tingting.zhu@eng.ox.ac.uk
@reference: 
GPy Library https://github.com/SheffieldML/GPy
GPy Document https://gpy.readthedocs.io/en/deploy/

"""

#%% Gaussian process classification

#%% import libraries

import GPy
import numpy as np
from matplotlib import pyplot as plt

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


#%% draw the latent function value from a RBF kernel
k = GPy.kern.RBF(1, variance=10., lengthscale=0.1)
np.random.seed(1000)
X = np.linspace(0., 1., 100)[:, None]
f = np.random.multivariate_normal(np.zeros(100), k.K(X,X))

plt.figure(figsize=(20,10))
plt.plot(X, f, 'b-')
plt.title('latent function values');
plt.xlabel('$x$');plt.ylabel('$f(x)$')


#%% squash the latent function between [0, 1] using the probit link function

# define link function

# transfer f

#%% draw samples form a Bernoulli distribution with success probability equal
# to the transformed latent function



#%% Inference via GP using expectation propagation

# definea kernel to generate f


# using the core GP model to tailor the desired inference method and likelihood 


#%% Inference via GP using Laplace

# definea kernel to generate f

# using the core GP model to tailor the desired inference method and likelihood 


# both Laplace and EP might give you different result for the hyperparameters!

#%% predict using X_new
X_new = np.linspace(0., 1., 50)[:, None]








