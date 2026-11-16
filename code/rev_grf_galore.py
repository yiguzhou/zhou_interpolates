"""
revision GRF models
"""

import sys
sys.path.append("../")
from config import *
from figs_config import *
from interpmodules import helpers
import copy
import gstools as gs
import joblib
import pandas as pd
import pyvista as pv
import random

models = {'spherical': gs.covmodel.Spherical, 
        'exponential': gs.covmodel.Exponential,
        'gaussian': gs.covmodel.Gaussian,
        'cubic': gs.covmodel.Cubic,
        'stable': gs.covmodel.Stable,
        'matern': gs.covmodel.Matern,
        'hole': gs.covmodel.JBessel}

# setting up
sphere_mesh = joblib.load(projdir + '/data/fsLR-4k_sphere_grf.pkl')
sphere_mesh.clear_data()
ranges = [10, 25, 50, 75, 100] # correlation length in mm
percentages = [5, 10, 25, 50] # sampling density

# initiate instances for other models
random.seed(321)
Instances = {}

for ri,ran in enumerate(ranges):

    instances = {}
    
    for ind, mname in enumerate(models):
        
        l = helpers._convert_to_length(ran)
        
        mdl = models[mname](dim = 3, len_scale=l, var=1.0)
        srf = gs.SRF(mdl)

        
        seed = random.randint(0,100000)
        random_map = srf.mesh(sphere_mesh, points='points', seed=seed)
    
        instances[mname] = random_map

    Instances[ran] = pd.DataFrame(instances)


# load samples
sampl = joblib.load(projdir + f'/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{pct}.pkl')
kc, kv, uc, uv = sampl[1]