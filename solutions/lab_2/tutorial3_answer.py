# -*- coding: utf-8 -*-
"""

@author: Tingting Zhu
@contact: tingting.zhu@eng.ox.ac.uk
@reference: 
GPy Library https://github.com/SheffieldML/GPy
GPy Document https://gpy.readthedocs.io/en/deploy/

"""

#%% import libraries
import numpy as np
from matplotlib import pyplot as plt
import GPy
import scipy as sp


#%% Description of the tutorial

# In this tutorial we will look at using a Gaussian Process with a Poisson 
# likelihood to model count data, i.e., Y={0,1,2,3,..}

#%% Understanding count data and Poisson Distribution
# Poisson distribution: https://en.wikipedia.org/wiki/Poisson_distribution

K = np.arange(0, 50, 1)[:, None] # count
rates = np.arange(0, 50, 1)[:, None] # rate

#Make a matrix with PMF for (rate x count) combinations
rates_count = np.array([sp.stats.poisson(r).pmf(K) for r in rates]).reshape(rates.shape[0], K.shape[0])

rates_plot = [1,5,10,20]

#Plot each rate as a function of counts K
plt.figure(figsize=(20, 10))
for r in rates_plot:
    plt.plot(K, rates_count[r, :], label='rate = {}'.format(float(rates[r])))
plt.title('Poisson probability mass distribution for different rates')
plt.ylabel('Probability Mass Function (PMF)')
plt.xlabel('Counts/Frequency of occurance')
plt.legend()

Ks = [1,5,10,20]
plt.figure(figsize=(20, 10))
for k in Ks:
    plt.plot(rates, rates_count[:, k], label='K = {}'.format(int(K[k])))
plt.xlabel('Rate')
plt.ylabel('PMF')
plt.legend()

#%% Relationship between Poisson and Gaussian Distribution

# As the rate increases, the Poisson distribution (over discrete values) becomes 
# a Gaussian distribution (over continuous values)
small_rate = 1
K = np.arange(0, 50, 1)[:, None]
Kcont = np.linspace(0, 50, 100)[:, None]
gauss50 = sp.stats.norm(loc=small_rate, scale=np.sqrt(small_rate)).pdf(Kcont)
poisson50 = sp.stats.poisson(small_rate).pmf(K)
plt.figure(figsize=(20, 10))
plt.plot(K, poisson50, label='Poisson')
plt.plot(Kcont, gauss50, label='Gaussian')
plt.title('Gaussian and Poisson with small rate')
plt.ylabel('PDF/PMF')
plt.xlabel('Counts')
plt.legend()

# large rate analysis:
# now set the small rate to be 30 and plot counts vs. PDF/PMF and rate vs. PDF/PMF


#%% Using GP to approximate Poisson for 48-hour ED arrival

fs = 48 # sample rate 
f = 2 # the frequency of the signal
X = np.linspace(0,fs,fs)[:, None]
intensities = lambda x: 10+ 5*np.sin(1.5*np.pi*f*(x/fs)) + np.sin(0.6*x)
plt.plot(X, intensities(X))
plt.title('Real underlying intensities (hourly rate for ED arrival)')

np.random.seed(10)
Y =np.array([sp.random.poisson(intensity) for intensity in intensities(X)])
plt.figure(figsize=(20, 10))
plt.bar(X.squeeze(),Y.squeeze(), align='center', alpha=0.5)
plt.xlabel('Time')
plt.ylabel('ED arrival rate')
plt.title('Observed counts of ED arrival over 48 hours')

# The Poisson distribution only support for non-negative integers, 
# where a Guassian supports all real numbers (i.e., -infinity to + infinity).
# we cannot directly place a Gaussian process prior directly on the rate of 
# the Poisson as the rate has to be positive.

#%% Define kernel
kernel = GPy.kern.RBF(1, variance=1.0, lengthscale=1.0)

#%% Define the poisson_likelihood using GPy.likelihoods
poisson_likelihood = GPy.likelihoods.Poisson()

#%% Choose an inference method (hint: Laplace)
laplace_inf = GPy.inference.latent_function_inference.Laplace() 

#%% Construct a GP model using GPy.core.GP

m = GPy.core.GP(X=X, Y=Y, likelihood=poisson_likelihood, inference_method=laplace_inf, kernel=kernel)

m.plot()
plt.xlabel('Time')
plt.ylabel('Observed counts of ED arrival throughout 48 hours')

#%% optimise your model m
m.optimize()
m.optimize_restarts(5, robust=True)
print(m)

#%% plot your optimised model m on the count data
m.plot()
plt.xlabel('Time (hours)')
plt.ylabel('Observed counts of ED arrival throughout 48 hours')

#%% make prediction on X_new

X_new = np.linspace(0,24,100)[:, None]
#Predictive GP for log intensity mean and variance
f_mean, f_var = m._raw_predict(X_new)
f_upper, f_lower = f_mean + 2*np.sqrt(f_var), f_mean - 2.*np.sqrt(f_var)
plt.plot(X, intensities(X), '--r', linewidth=2, label='true intensity')
#Plotting Y on an exponential scale as we are now looking at intensity rather than log intensity
plt.plot(X_new, np.exp(f_mean), lw=2)
plt.fill_between(X_new[:,0], np.exp(f_lower[:,0]), np.exp(f_upper[:,0]), alpha=.1)
plt.title('Real intensity vs posterior estimation')
plt.xlabel('Time (hours)')
plt.ylabel('Intensity')
plt.legend()

#%% Alternative
X_new = np.linspace(0,24,100)[:, None]
mean, Cov = m._raw_predict(X_new)

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

# Plot the GP fit mean and covariance
plot_gp(X_new, np.exp(mean), Cov, training_points=(X,Y))
plt.plot(X_new, intensities(X_new),"r:",lw=5)
plt.title("GPy regression model fit");



