'''
SWR fine-tuning
'''

import numpy as np
import copy
from config import *
from figs_config import *
from interpmodules import helpers
import joblib
import lapy
from lapy import TriaMesh
from lapy.solver import *
import matplotlib.pyplot as plt
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW
import nibabel as nb
import neuromaps.datasets
import numpy as np
import pandas as pd
import pyvista as pv
from scipy.stats import zscore
from sklearn.metrics import root_mean_squared_error
import time

def _learn_gw_regressor(X0, Y0, Yk0, bandwidth, spherical=True):
    '''
    Yk0: covariates measured at sampled location
    yk1: covariates measured at target location

    LEARN GEOGRAPHICALLY WEIGHTER REGRESSOR ONE SHOT
    REUSE REGRESSOR FOR INTERPOLATION AT EVERY POINT - MUCH faster

    OLS INITIATED AT EVERY LOCATION i ACROSS n points
    LEARN COEFFICIENTS FOR EVERY COVARIATE k ACROSS p predictors

    KERNEL BEING BISQUARE WITH ADAPTIVE NEIGHBOURHOOD FOR MAKING WEIGHTS (binary)
    
    '''

    Lon, Lat, Rho = helpers._cartesian_to_spherical(X0, unit='degrees')
    Sph = np.vstack([Lon, Lat])

    # Reformat coordinates array to make the libpysal function happy
    coords = list(zip(Sph[0,:], Sph[1,:]))
    y = Y0.reshape(-1,1)
    X = Yk0.T

    # get bandwith (aka adaptive neighbourhood size) for regressions
    if bandwidth == None:
        selector = Sel_BW(coords, y, X_loc=X, fixed=False, spherical=spherical)
        
        try:
            bandwidth = selector.search(criterion='AICc', search_method = 'golden_section') # minimize estimation error using spatial information from sampled locations
            # as opposed to CV minimizing prediction error in subsets of sampled location - not the most useful

        except np.linalg.LinAlgError:
            bandwidth = 100.0
    print(bandwidth)
    regressor = GWR(coords, y, X, bw=bandwidth, kernel='bisquare', spherical=spherical, fixed=False) # Wheeler and Paez, 2010
    
    return regressor, selector



def interpolate_SWR(X0, Y0, Yk0, x1, yk1, bandwidth=None, spherical=True, timeit=False):
    '''
    iteratively train a regressor across query points
    '''
    
    if timeit: t1 = time.time()
        
    reg, sel = _learn_gw_regressor(X0, Y0, Yk0, bandwidth, spherical=True)

    mdl = reg.fit()
    scale = mdl.scale
    residuals = mdl.resid_response

    Lon, Lat, Rho = helpers._cartesian_to_spherical(x1, unit='degrees')
    Sph = np.vstack([Lon, Lat])

    # Reformat coordinates array to make the libpysal function happy
    pred_coords = np.asarray(list(zip(Sph[0,:], Sph[1,:])))

    pred_X = yk1
    pred_results = reg.predict(pred_coords, pred_X.T, scale, residuals)

    y1_interp = pred_results.predictions.flatten()
    
    if timeit: t2 = time.time(); return y1_interp, t2 - t1

    return y1_interp

'''
def compute_lbo(geometry):
    
    assuming geometry is a TriaMesh instance
    
    # Compute FEM matrices under homogeneous LBO
    solver = lapy.Solver(geometry)
    stiffness, mass = solver._fem_tria(tria=geometry)
    return stiffness, mass
'''
def eigendecompose(data, geometry, n_modes, v0=None, seed=321):
     solver = lapy.Solver(geometry)
     evals, emodes = solver.eigs(k=n_modes, v0=v0, rng=seed)

     return evals, emodes

#--------set up--------#
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")

# eigendecomposition of the sphere
path = neuromaps.datasets.fetch_fslr(density = '4k')['sphere'].L
gii = nb.load(path)

vertices=gii.darrays[0].data
faces=gii.darrays[1].data

tmesh = TriaMesh(v=vertices, t=faces)

ran=50
pct=25
n_modes = 500

evals, emodes = eigendecompose(data= sphere_mesh[str(ran)], geometry = tmesh, n_modes=n_modes)

print(evals.shape)
print(emodes.shape)

#pl = pv.Plotter(shape = (1, 6), window_size = (400*6, 400))


#for i, mi in enumerate([1, 10, 50, 100, 200, 500]):
#    pl.subplot(0,i)
#    tmp_mesh = copy.deepcopy(sphere_mesh)
#    pl.add_mesh(tmp_mesh, scalars = emodes[:, mi-1], cmap = bpr, **mesh_aes)
#    pl.add_text(f"m={str(mi)}", font = "courier", font_size = 10, position = "lower_edge")
#    pl.camera_position = "yz"
#pl.save_graphic(projdir + "/figs/swr_eigenmodes.svg")


#--------data configuration--------#
sampl = joblib.load(projdir + f"/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{pct}.pkl")
kc, kv, uc, uv = sampl[1]

data = zscore(sphere_mesh[str(ran)])

sparse_coords = kc[:, :, 0]
sparse_data = data[kv.flatten().astype(int)]
test_coords = uc[:, :, 0]

Nkeep = [5, 10, 50]

scores = []

for nkeep in Nkeep:
    train_covariates = emodes[kv.flatten().astype(int), :nkeep]
    test_covariates = emodes[uv.flatten().astype(int), :nkeep]

    #nfolds = 5
    #Cv_pred = []
    #cv_kc, cv_ki, cv_uc, cv_ui = get_cv_splits(sparse_coords, nfolds = nfolds, rseed=321)

    print(test_coords.shape)
    print(sparse_coords.shape)
    print(test_covariates.T.shape)

    args = {"X0": sparse_coords,
            "Y0": sparse_data,
            "Yk0": train_covariates.T,
            "x1": test_coords,
            "yk1": test_covariates.T
            }

    y_pred = interpolate_SWR(**args)
    print(y_pred.shape)
    y_true = np.asarray(data[uv.flatten().astype(int)])

    nrmse = root_mean_squared_error(
        y_true = y_true,
        y_pred=y_pred
        ) / np.nanstd(y_true)

    scores.append(nrmse)

print(scores)