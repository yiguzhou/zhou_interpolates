'''
Evaluate noise
'''

from config import *
import copy
from figs_config import *
import joblib
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.stats import zscore


NOISE_LEVELS = [1, 5, 10, 15, 20, 50]


#--------helper functions--------#
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


#--------set up GRF plus noise--------#
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")

ranges = [25, 50, 75, 100]

plot = False
if plot:
    pl = pv.Plotter(shape = (len(ranges) + 1, len(NOISE_LEVELS) + 1), window_size = ((len(NOISE_LEVELS) + 1)*400, (len(ranges) + 1) * 400))

    for ri, ran in enumerate(ranges):

        data = zscore(sphere_mesh[str(ran)])

        noisy_data = dict.fromkeys(NOISE_LEVELS)
        noisy = dict.fromkeys(NOISE_LEVELS)
        for noise_level in NOISE_LEVELS:
            noise = zscore(make_some_noise(sphere_mesh, sigma = noise_level))
            add_noise = data + noise
            noisy[noise_level] = noise
            noisy_data[noise_level] = add_noise

        pl.subplot(ri + 1, 0)
        tmp_mesh = copy.deepcopy(sphere_mesh)
        pl.add_mesh(tmp_mesh, scalars = str(ran), cmap = bpr, **mesh_aes)
        pl.add_text(f"GRF({ran}mm)", position = "lower_edge", font = "courier", font_size = 10)
        pl.camera_position = "yz"

        for i, noise_level in enumerate(NOISE_LEVELS):

            pl.subplot(0, i + 1)
            tmp_mesh = copy.deepcopy(sphere_mesh)
            pl.add_mesh(tmp_mesh, scalars = noisy[noise_level], cmap = "gist_yarg", **mesh_aes)
            pl.add_text(f"W+sigma={noise_level}", position = "lower_edge", font = "courier", font_size = 10)
            pl.camera_position = "yz"

            pl.subplot(ri + 1, i + 1)
            tmp_mesh = copy.deepcopy(sphere_mesh)
            pl.add_mesh(tmp_mesh, scalars = noisy_data[noise_level], cmap = bpr, **mesh_aes)
            pl.add_text(f"+sigma={noise_level}", position = "lower_edge", font = "courier", font_size = 10)
            pl.camera_position = "yz"

    pl.save_graphic(projdir + "/figs/noisy_grf_examples.svg")

#--------run--------#