'''
RBF hyperparameter tuning
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

# declare the parameter search grid
PARAM_GRID = {
              "kernel": np.asarray([
                  "linear",
                  "thin_plate_spline",
                  "cubic",
                  "quintic",
                  "multiquadratic",
                  "inverse_multiquadratic",
                  "inverse_quadratic",
                  "gaussian"]),
              "n_neighbours": np.linspace(2, 20, 19)
}

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
initial = int(np.prod(np.asarray([len(PARAM_GRID[elem]) for elem in PARAM_GRID])))
print(initial)

ran=100
pct=50

#ranges = [25, 100]
#percentages = [5, 10, 25, 50]

interp = interpolate_RBF

sampl = joblib.load(projdir + f"/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{pct}.pkl")
kc, kv, uc, uv = sampl[1]

data = zscore(sphere_mesh[str(ran)])

sparse_coords = kc[:, :, 0]
sparse_data = data[kv.flatten().astype(int)]
test_coords = uc[:, :, 0]

nfolds = 5
Cv_pred = []
cv_kc, cv_ki, cv_uc, cv_ui = get_cv_splits(sparse_coords, nfolds = nfolds, rseed=321)


kern_is = range(len(PARAM_GRID["kernel"]))
print(kern_is)
n_js = range(len(PARAM_GRID["n_neighbours"]))
print(n_js)

best_score = 1
best_params = (
    PARAM_GRID["kernel"][kern_is[0]],
    PARAM_GRID["n_neighbours"][n_js[0]]
)

NRMSE = np.ones((len(PARAM_GRID["kernel"]), len(PARAM_GRID["n_neighbours"])))
Nrmse = np.empty((len(PARAM_GRID["kernel"]), len(PARAM_GRID["n_neighbours"]), nfolds))
Nrmse[:] = np.nan

for i in kern_is:
    for j in n_js:
        for ind in range(nfolds):

            #print("inverse distance power: " + str(PARAM_GRID["power"][pow_is[n]]))
            #print("number of neighbours: " + str(PARAM_GRID["n_neighbours"][n_js[n]]))

            args = {"X0": cv_kc[ind],
                "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                "x1": cv_uc[ind],
                "kernel": PARAM_GRID["kernel"][i],
                "n_neighbours": PARAM_GRID["n_neighbours"][j]}

            try:
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
                
                Nrmse[i, j, ind] = nrmse
            except ValueError:
                Cv_pred.append(np.nan)
                Nrmse[i, j, ind] = np.nan

        score = np.nanmean(Nrmse[i, j, :])
        NRMSE[i, j] = score
                
        if best_score > score:
            best_score = score
            best_params = (
            PARAM_GRID["kernel"][i],
            PARAM_GRID["n_neighbours"][j]
            )
print(best_score)
print(best_params)
'''
fig, ax = plt.subplots(figsize = (10, 10))
sns.heatmap(
    NRMSE.T, 
    annot = NRMSE.T,
    square=True, 
    vmax = 1,
    vmin=0, 
    cmap = "gist_yarg",
    linewidth = 1,
    linecolor = 'w',
    ax=ax)


row_labels = []

data = pd.DataFrame(NRMSE.T, 
                    columns = PARAM_GRID["kernel"], 
                    index = PARAM_GRID["n_neighbours"])

col_label = np.where( data == np.nanmin(data) )[0]
row_label = np.where( data == np.nanmin(data) )[1]

i = row_label[0]
j = col_label[0]


print(i)
print(j)
ax.add_patch(
    FancyBboxPatch(
        (i, j), 1, 1,
        boxstyle="round,pad=0.0001",
        linewidth=2,
        edgecolor="red",
        facecolor="none", clip_on=False
    ))
    
plt.tight_layout()
plt.show()
'''
#--------test--------#
kernel = best_params[0]
neighbours = best_params[1]

args = {"X0": sparse_coords,
        "Y0": sparse_data,
        "x1": test_coords,
        "kernel": kernel,
        "n_neighbours": neighbours}

pred = interp(**args)

full_map = np.zeros(sphere_mesh.n_points)
full_map[kv.flatten().astype(int)] = sparse_data
full_map[uv.flatten().astype(int)] = pred

pl = pv.Plotter(shape=(1,2), window_size = (800, 400))

pl.subplot(0,0)
tmp_mesh = copy.deepcopy(sphere_mesh)
pl.add_mesh(tmp_mesh, scalars = data)
pl.add_text(f"{ran}mm", font = 'courier', font_size = 10, position = "lower_edge")
pl.camera_position = "yz"
pl.subplot(0,1)
tmp_mesh = copy.deepcopy(sphere_mesh)
pl.add_mesh(tmp_mesh, scalars = full_map)
pl.add_text(f"{kernel} at {neighbours}nbs", font = 'courier', font_size = 10, position = "lower_edge")
pl.camera_position = 'yz'

pl.save_graphic(projdir + "/figs/rev_rbf_example.svg")