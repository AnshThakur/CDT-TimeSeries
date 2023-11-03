# -*- coding: utf-8 -*-
"""

@author: Tingting Zhu
@contact: tingting.zhu@eng.ox.ac.uk
@reference: 
GPy Library https://github.com/SheffieldML/GPy
GPy Document https://gpy.readthedocs.io/en/deploy/

"""


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


#%% squash the latent function between [0, 1] using the Bernoulli - EXTRA
lik = GPy.likelihoods.Bernoulli()
p = lik.gp_link.transf(f) # squash the latent function
plt.figure(figsize=(20,10))
plt.plot(X, p, 'r-')
plt.title('latent probabilities');plt.xlabel('$x$');plt.ylabel('$\sigma(f(x))$')

#%% squash the latent function between [0, 1] using the probit link function
# define link function
probit = GPy.likelihoods.link_functions.Probit()
# transfer f
p_s = probit.transf(f)
plt.figure(figsize=(20,10))
plt.plot(X, p_s, 'r-')
plt.title('latent probabilities');plt.xlabel('$x$');plt.ylabel('$\sigma(f(x))$')

#%% draw samples form a Bernoulli distribution with success probability equal
# to the transformed latent function
y = GPy.likelihoods.Bernoulli(gp_link=probit).samples(f)[:, None]

# Plot the probit squashed f and samples from the corresponding Bernoulli distribution
plt.figure(figsize=(20, 10))
plt.plot(X, y, 'kx', X, p_s, "-", mew=3)
plt.xlabel("$x$"), plt.ylabel("$y$/$p(y=1)$")
plt.legend(labels=["Binary observations", "Squashed latent function: $\Phi(f(\cdot))$"], loc='center right')
plt.title("Observations sampled from Bernoulli distributions with probability $\Phi(f(\cdot))$");

#%% Inference via GP using expectation propagation

# definea kernel to generate f
k = GPy.kern.RBF(1, variance=1., lengthscale=0.2)

# using the core GP model to tailor the desired inference method and likelihood 
m = GPy.core.GP( 
    X=X, 
    Y=y,        
    kernel = k,  
    inference_method = GPy.inference.latent_function_inference.expectation_propagation.EP(), 
    likelihood = GPy.likelihoods.Bernoulli(gp_link=probit)
)
m.optimize()
m.optimize_restarts(5, robust=True)
print(m)

#%% Inference via GP using Laplace

# definea kernel to generate f
k = GPy.kern.RBF(1, variance=1., lengthscale=0.2)

# using the core GP model to tailor the desired inference method and likelihood 
m = GPy.core.GP( 
    X=X, 
    Y=y,        
    kernel = k,  
    inference_method = GPy.inference.latent_function_inference.Laplace(), 
    likelihood = GPy.likelihoods.Bernoulli(gp_link=probit)
)
m.optimize()
m.optimize_restarts(5, robust=True)
print(m)

# both Laplace and EP might give you different result for the hyperparameters!

#%% predict using X_new
X_new = np.linspace(0., 1., 50)[:, None]

mean, Cov = m.predict(X_new, include_likelihood=False, full_cov=True)
plt.figure(figsize=(20,10))
plt.subplot(211)
plot_gp(X_new, mean, Cov)
plt.plot(X, f, "--",lw=5)
plt.title("GP fit of latent function with Bernoulli likelihood")
plt.legend(labels=["samples from GP posterior","true $f(x)$"]);

# We will also predict the median and 95% confidence intervals of the likelihood
quantiles = m.predict_quantiles(X_new, quantiles=np.array([50.]), likelihood=m.likelihood)
prob, _ = m.predict(X_new, include_likelihood=True) # Probability function for Bernoulli

plt.subplot(212)
plt.plot(X_new, quantiles[0], "--", X_new, prob, "--",lw=3)
plt.plot(X, y, "kx", mew=2)
plt.xlabel("$x$"), plt.ylabel("$p(y=1)$")
plt.title("Classifier prediction");
plt.legend(labels=["median GP likelihood", "Bernoulli probability", "training points"]);






