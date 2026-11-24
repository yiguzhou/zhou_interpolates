'''
checking multiquadric kernels
'''

from config import *
import copy
import joblib
from interpmodules import helpers
import itertools
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import pyvista as pv
import random
from scipy.interpolate import RBFInterpolator
from scipy.spatial import KDTree
from scipy.stats import zscore
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
import time

# load data
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")

#--------helper functions--------#
def get_cv_splits(coords, nfolds=5, rseed=321):

    np.random.seed(rseed)
    ncoords = coords.shape[0]

    n_test = int( 1/nfolds * ncoords )
    print(n_test)
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


def interpolate_RBF(X0, Y0, x1, n_neighbours=10, kernel='cubic', timeit=False):
    '''
    wrapper for scipy interpolator

    others are available through scipy
    '''
    
    if timeit: t1=time.time()
    
    y1 = np.full(x1.shape[0], np.nan)

    for eps in range(1,6):

        try:
            interpolator = RBFInterpolator(y=X0, d=Y0, neighbors=n_neighbours, \
                            kernel=kernel, epsilon=eps, smoothing=1e-4) # small scaling of smoothing parameter for very sparse samples
            y1 = interpolator(x1);

        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err): continue

    if timeit: t2=time.time(); return y1,t2-t1
    
    return y1

#--------start searching (exhaustive grid search)--------#

ran=75
pct=5
kernels = ["multiquadric", "inverse_multiquadric"]
n_neighbours = np.linspace(2, 20, 19)

interp = interpolate_RBF

sampl = joblib.load(projdir + f"/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{pct}.pkl")
kc, kv, uc, uv = sampl[1]

data = zscore(sphere_mesh[str(ran)])

sparse_coords = kc[:, :, 0]
sparse_data = data[kv.flatten().astype(int)]
test_coords = uc[:, :, 0]

#--------set up cross validation-------#
nfolds = 5
Cv_pred = []
cv_kc, cv_ki, cv_uc, cv_ui = get_cv_splits(sparse_coords, nfolds = nfolds, rseed=321)


#--------start computing scores--------#
best_score = 1
best_params = (kernels[1], 0)

Nrmse = np.empty((len(n_neighbours), nfolds))
Nrmse[:] = np.nan
Meanscores = np.empty(len(n_neighbours))
Meanscores[:] = np.nan

for i, n in enumerate(n_neighbours):

    # cross-validation
    for ind in range(nfolds):

        args = {"X0": cv_kc[ind],
            "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
            "x1": cv_uc[ind],
            "kernel": kernels[1],
            "n_neighbours": n}

        cv_pred = interp(**args)
        print(cv_pred)
        Cv_pred.append(cv_pred)

        cv_true = np.asarray(sparse_data[cv_ui[ind].flatten().astype(int)])
        print(cv_true)
        print(np.nanstd(cv_true))
        nrmse = root_mean_squared_error(
            y_true = cv_true,
            y_pred=cv_pred
            ) / np.nanstd(cv_true)
        
        Nrmse[i, ind] = nrmse


    score = np.nanmean(Nrmse[i, :])
    Meanscores[i] = score
            
    if best_score > score:
        best_score = score
        best_params = (
        kernels[1],
        n
        )
print(best_score)
print(best_params)

fig, ax = plt.subplots(figsize = (5,3))
df = pd.DataFrame(Nrmse)
print(df.head())
df_long = df.reset_index(names = "n_neighbours").melt(
    id_vars = "n_neighbours",
    var_name = "fold",
    value_name = "nrmse"
)
sns.lineplot(df_long, x = "n_neighbours", y ="nrmse")
plt.tight_layout()
plt.show()