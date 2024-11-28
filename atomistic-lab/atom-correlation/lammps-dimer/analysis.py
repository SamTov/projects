import numpy as np 

import chemfiles
from atomcorr.analysis.onsager_coefficients import OnsagerCoefficients
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import mdsuite as mds
from mdsuite.utils import Units

import h5py as hf

def compute_atom_correlation(data):
    """
    Compute atom-wise correlation tensor.
    """    
    calculator = OnsagerCoefficients()

    correlation_matrix = calculator._compute_correlation_matrix(
        data,
        data,
        correlation_time=500,
        data_range=500
    )

    return correlation_matrix


def normalize_covariance_matrix(covariance_matrix: np.ndarray):
    """
    Method for normalizing a covariance matrix.
    Returns
    -------
    normalized_covariance_matrix : np.ndarray
            A normalized covariance matrix, i.e, the matrix given if all of its inputs
            had been normalized.
    """
    order = np.shape(covariance_matrix)[0]

    diagonals = np.diagonal(covariance_matrix)

    repeated_diagonals = np.repeat(diagonals[None, :], order, axis=0)

    normalizing_matrix = np.sqrt(repeated_diagonals * repeated_diagonals.T)

    return covariance_matrix / normalizing_matrix


with hf.File("water-lammps/lj/database.hdf5") as db:

    velocity = np.concatenate(
        (
            np.transpose(db['1']["Velocities"][:], (1, 0, 2)), 
            np.transpose(db['2']["Velocities"][:], (1, 0, 2))
        ), 
        axis=1
    )

matrix = compute_atom_correlation(velocity)

np.save("matrix.npy", matrix)