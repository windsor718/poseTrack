"""
coded by: Yuta Ishitsuka

bayesian update toolkits
"""
import sys
import os
import numpy as np

def initialize(points):
    probs = np.ones([len(points)])*(1/len(points))
    return probs

def update(priors, likelihoods):
    """
    calculates posterior probability based on Bayesian update.

    Args:
        priors (np.ndarray): prior probability array
        liklihoods (np.ndarray): likelihoods array

    Returns:
        np.ndarray: posterior probability array
    """
    posteriors = priors*likelihoods/(priors*likelihoods).sum()
    #print("priors", priors)
    #print("likelihood", likelihoods)
    #print("posteriors", posteriors)
    return posteriors

def gausianLikelifood(distances, scale="auto", sigma=100000):
    """
    calculates likelihood probability based on the gausian distribution
    with a given scale.

    Args:
        disatances (np.ndarray): distance array
        scale (float): vertical scaling for the distribution

    Returns:
        np.ndarray: likelihood probability array
    """
    if scale=="auto":
        scaler = (np.sqrt(2*np.pi*sigma))**(-1)*np.exp(-(0)**2/(2*sigma))
        scale = 1/scaler
    likelihoods = scale*((np.sqrt(2*np.pi*sigma)) \
                    **(-1)*np.exp(-(distances)**2/(2*sigma)))
    likelihoods = np.maximum(likelihoods, 1e-8)
    return likelihoods

def bayesianUpdate(priors, distances):
    """
    update prior probability into posterior by Bayesian update.

    Args:
        priors (np.ndarray): prior probabilities
        distances (np.ndarray): distance array

    Returns:
        np.ndarray: posteripor probabilities
    """
    likelihoods = gausianLikelifood(distances)
    posteriors = update(priors, likelihoods)
    return posteriors
