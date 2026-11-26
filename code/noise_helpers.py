'''
noise analysis helper functions
'''
from config import *
from figs_config import *
import numpy as np
from scipy.spatial import cKDTree

NOISE_LEVELS = [1, 5, 10, 15, 20, 50]


#--------helper functions--------#

#====for making nonstationary maps====#
def linear_scaling(x0, lims0, lims1):

    x1 = lims1[0] + (np.asarray(x0) - lims0[0]) * (lims1[1] - lims1[0]) / (lims0[1] - lims0[0])
    
    return x1

def gaussian_gradual_blur(mesh, values, sigma_i):
    """
    uneven blurring
    """

    radius = np.max(mesh.points) * 2

    points = np.asarray(mesh.points)
    tree = cKDTree(points)
    neighborhoods = tree.query_ball_point(points, radius)

    smoothed = np.zeros(values.shape)

    # calculate gaussian weighted average
    for i, neighbors in enumerate(neighborhoods):

        sigma = linear_scaling(points[i][0], [-100,100], [1,20])
        print(sigma)

        neighbors = np.asarray(neighbors)
        distances = np.linalg.norm(
            points[neighbors] - points[i],
            axis=1
        )
        weights = np.exp(
            -0.5 * (distances / sigma) ** 2
        )
        weights /= weights.sum()
        smoothed[i] = np.tensordot(
            weights,
            values[neighbors],
            axes=(0, 0)
        )

    return smoothed


#====for making noise====#
def white(mesh, seed = 321):
    from numpy.random import default_rng

    rng = default_rng(seed)
    w = rng.normal(loc=0, scale=1, size=mesh.n_points)
    return w

def gaussian_blur_point_data(mesh, values, sigma, radius=None):
    """
    blur noise
    """

    if radius is None:
        radius = 3.0 * sigma

    points = np.asarray(mesh.points)
    tree = cKDTree(points)
    neighbourhoods = tree.query_ball_point(points, radius)

    smoothed = np.zeros(values.shape)

    # calculate gaussian weighted average
    for i, neighbours in enumerate(neighbourhoods):

        neighbours = np.asarray(neighbours)
        distances = np.linalg.norm(
            points[neighbours] - points[i],
            axis=1
        )
        weights = np.exp(
            -0.5 * (distances / sigma) ** 2
        )
        weights /= weights.sum()
        smoothed[i] = np.tensordot(
            weights,
            values[neighbours],
            axes=(0, 0)
        )

    return smoothed

def make_some_noise(mesh, sigma):
    '''
    sigma being how much blur
    '''
    W = white(mesh)
    blurred_noise = gaussian_blur_point_data(
        mesh = mesh,
        values = W,
        sigma = sigma
        )
    return blurred_noise

#====for cross-validation====#
def get_cv_splits(coords, nfolds=5, rseed=321):

    np.random.seed(rseed)
    ncoords = coords.shape[0]

    n_test = int( 1/nfolds * ncoords )
    cv_test_samples = []

    available_ind = np.arange(ncoords)

    for ind in range(nfolds):
        if ind == nfolds - 1:
            cv_test_samples.append(available)
            break

        random_sample = np.random.choice(available_ind, size = n_test, replace=False)
        cv_test_samples.append(random_sample)
        available = np.setdiff1d(available_ind, random_sample)

    
    cv_test_inds = cv_test_samples
    cv_test_coords = tuple( [ coords[test_vert_set,:] for test_vert_set in cv_test_inds ] )
    cv_train_inds= tuple( [ np.setdiff1d(np.arange(ncoords), test_vert_set) for test_vert_set in cv_test_inds] )
    cv_train_coords = tuple( [ coords[train_vert_set, :] for train_vert_set in cv_train_inds] )
    return cv_train_coords, cv_train_inds, cv_test_coords, cv_test_inds

#====for step 2 of KNN parameter selection====#
def indices_from_radial_nb(i, j, size=100):
    from math import sqrt
    radius = int( 0.5 * (sqrt(size) - 1))

    i_s = [ i + row_dist for row_dist in np.linspace(-radius, radius, 2*radius+1) ]
    j_s = [ j + col_dist for col_dist in np.linspace(-radius, radius, 2*radius+1) ]

    return i_s, j_s
