import numpy as np
import random   
from numpy.linalg import norm
from numpy.random import randint
from numpy import linalg as LA
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import math

import sys
if sys.version_info >= (3,0):
    from functools import reduce

def origin_mass_center(pcd):
    expectation = np.mean(pcd, axis = 0)
    centered_pcd = pcd - expectation
    return centered_pcd

def resample_pcd(pcd, n):
    """Drop or duplicate points so that pcd has exactly n points"""
    idx = np.random.permutation(pcd.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.choice(pcd.shape[0], size = n - pcd.shape[0])])
    return pcd[idx[:n]]

def normalize(points, unit_ball = False):
    normalized_points = origin_mass_center(points)
    l2_norm = LA.norm(normalized_points,axis=1)
    max_distance = max(l2_norm)
    if unit_ball:
        normalized_points = normalized_points/(max_distance)
    else:
        normalized_points = normalized_points/(2 * max_distance)

    return normalized_points

def show_pcd(X_iso, show=True):
    lim = 0.8
    fig = pyplot.figure()
    ax = Axes3D(fig)
    ax.set_xlim([-lim, lim])
    ax.set_ylim([-lim, lim])
    ax.set_zlim([-lim, lim])
    sequence_containing_x_vals = list(X_iso[:, 0])
    sequence_containing_y_vals = list(X_iso[:, 1])
    sequence_containing_z_vals = list(X_iso[:, 2])

    C = np.array(['#ff0000' for i in range(len(X_iso))])
    ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c=C, s=10, depthshade=True) 
    ax.grid(False)
    ax.axis(False)
    if show:
        pyplot.show()

def make_holes_pcd(pcd, hole_size=0.1):
    """
    Arguments:
        pcd {[float[n,3]]} -- [point cloud data of n size in x, y, z format]

    Returns:
        [float[m,3]] -- [point cloud data in x, y, z of m size format (m < n)]
    """
    rand_point = pcd[randint(0, pcd.shape[0])]
    partial_pcd = []
    hole_pcd = []
    for i in range(pcd.shape[0]):
        dist = np.linalg.norm(rand_point - pcd[i])
        if dist >= hole_size:
            partial_pcd.append(pcd[i])
        else:
            hole_pcd.append(pcd[i])

    return np.array(partial_pcd), np.array(hole_pcd)


def make_holes_pcd2(pcd, hole_size=0.1, n_partial=2048):
    """[summary]
    Arguments:
        pcd {[float[n,3]]} -- [point cloud data of n size in x, y, z format]

    Returns:
        [float[m,3]] -- [point cloud data in x, y, z of m size format (m < n)]
    """
    rand_point = pcd[randint(0, pcd.shape[0])]
    dist = np.linalg.norm(pcd - rand_point, axis=-1)
    idx = np.argsort(-dist)
    k = sum(dist > hole_size)
    if k > n_partial:
        idx[:k] = idx[np.random.permutation(np.arange(k))]

    return pcd[idx[:n_partial]], pcd[idx[max(k, n_partial):]]