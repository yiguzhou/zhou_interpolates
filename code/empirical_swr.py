'''
swr covariate tuning empirical
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


#--------two different sets of covariates--------#
# geometric eigenmodes


# empirical sets
def choose_empirical_covariates(mesh, curr_map, ntops=10):
        
        other_maps = mesh.array_names

        CMAT = np.ones(len(other_maps))

        for j, other_map in enumerate(other_maps):
            
            if other_map == curr_map: continue

            CMAT[j] = np.ma.corrcoef(np.ma.masked_invalid(mesh[curr_map]), \
                                np.ma.masked_invalid(mesh[other_map]))[1,0]

        
        chosen_map_idx = abs(CMAT).argsort()

        chosen_map_idx = chosen_map_idx[:-1]

        top_cov = np.array(other_maps)[chosen_map_idx][-5:]
        print(curr_map)
        print(top5_cov)
        print(CMAT[chosen_map_idx[-5:]])

        X_covariates = np.vstack([src_polydata[top5_cov_i] for top5_cov_i in top5_cov])
        Yk0 = X_covariates

        if trg_polydata == None: return Yk0
        
        yk1 = np.vstack([trg_polydata[top5_cov_i] for top5_cov_i in top5_cov])

        return Yk0, yk1
    return Yk0, yk1
#--------set up--------#
reduced_mesh = joblib.load(projdir + "/data/fsLR-4k_reduced_midthickness_neuromaps.pkl")