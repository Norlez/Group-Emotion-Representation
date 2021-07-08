#!/usr/bin/env python3

import numpy as np
import random

def softmax(z):
    """
    Turns a vector of real values to another vector of real values whose sum is 1. 

    Parameters
    ----------
    z : numpy.ndarray
        real valued vector
        
    Returns
    -------
    numpy.ndarray
        Softmax of the input vector.
    """
    return np.exp(z) / np.sum(np.exp(z))


def update_probability(current_probability, transition_matrix):
    """
    Updates the probability by one step using a Markov transition matrix.
    
    Parameters
    ----------
    current_probability : numpy.ndarray
        current probability of the variables
    transition_matrix : numpy.ndarray
        Markov transition matrix
    
    Returns
    -------
    numpy.ndarray
        probability updated by one step
    """
    return np.dot(current_probability, transition_matrix)


def change_transition_matrix(transition_matrix, change_probability=0.2):
    """
    Randomly modifies the Markov transition matrix (row-wise).
    
    Parameters
    ----------
    transition_matrix : numpy.ndarray
        Markov transition matrix
    change_probability : float
        probability that one or more rows of the transition matrix changes randomly.
    
    Returns
    -------
    transition_matrix : numpy.ndarray
        the updated transition matrix
    """
    for i in range(3):
        if random.uniform(0, 1) > (1.0 - change_probability):
            row = np.array([random.randint(0, 10), random.randint(0, 10), random.randint(0, 10)])
            transition_matrix[i, :] = softmax(row)
    return transition_matrix


