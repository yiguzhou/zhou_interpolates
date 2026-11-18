'''
KNN parameter tuning
- we use successive halving to find an optimal parameter for interpolation
'''

from config import *
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
from scipy.spatial import KDTree
from scipy.stats import zscore
import seaborn as sns
from sklearn.metrics import root_mean_squared_error
import time


# load data
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")

# declare the parameter search grid
PARAM_GRID = {"n_neighbours": np.linspace(2, 20, 19),
              "power": np.linspace(0.1, 20, 200)
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

def get_inverse_distance_weights(dist, power):
    
    with np.errstate(divide='ignore'):
        weights = 1. / dist ** power
    isinf = np.isinf(weights)
    infrow = np.where(isinf)
    weights[infrow] = isinf[infrow]

    return weights

def interpolate_KNN(X0, Y0, x1, n_neighbours=5, power=2, timeit=False):
    '''
    Find nearest n-many neighbours, average their values
    and weigh inversely by distance abagen-style
    this is euclidean distance
    '''
    
    if timeit:
        t1=time.time()
        
    tree= KDTree(np.c_[X0[:,0], X0[:,1], X0[:,2]])
    dist, idx = tree.query(x1, k=n_neighbours)

    w = helpers._get_inverse_distance_weights(dist, power)

    # get average of nearest neighbors
    y1 = np.sum(Y0[idx]* w, axis=1) / np.sum(w, axis = 1)

    if timeit: t2=time.time(); return y1,t2-t1
    
    return y1


# implement search strategy

initial = int(np.prod(np.asarray([len(PARAM_GRID[elem]) for elem in PARAM_GRID])))
print(initial)
#ran=50
ranges = [25]
#pct=5
percentages = [5, 10, 25, 50]

interp = interpolate_KNN


#NRMSE = {}

#random.seed(321)
#pow_is = [ random.randint(0, len(PARAM_GRID["power"]) - 1) for _ in range(initial) ]
#print(pow_is)
#n_js = [ random.randint(0, len(PARAM_GRID["n_neighbours"]) - 1) for _ in range(initial) ]
#print(n_js)

pow_is = list(range(initial))
print(pow_is)
n_js = list(range(initial))
print(n_js)

#fig, axs = plt.subplots(2, 4, figsize = (40, 20), sharex=True, sharey=True)
fig, axs = plt.subplots(1,1,figsize = (200, 50))
#for ri, ran in enumerate(ranges):
ri=0
pct_i=0
ran=50
pct=25
#for pct_i, pct in enumerate(percentages):

sampl = joblib.load(projdir + f"/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{pct}.pkl")
kc, kv, uc, uv = sampl[1]

data = zscore(sphere_mesh[str(ran)])

sparse_coords = kc[:, :, 0]
sparse_data = data[kv.flatten().astype(int)]
test_coords = uc[:, :, 0]


nfolds = 10
Cv_pred = []

cv_kc, cv_ki, cv_uc, cv_ui = get_cv_splits(sparse_coords, nfolds = nfolds, rseed=321)

best_score = 1
best_params = (
    PARAM_GRID["power"][pow_is[0]],
    PARAM_GRID["n_neighbours"][n_js[0]]
)

NRMSE = np.ones((len(PARAM_GRID["power"]), len(PARAM_GRID["n_neighbours"])))
Nrmse = np.empty((len(PARAM_GRID["power"]), len(PARAM_GRID["n_neighbours"]), nfolds))
Nrmse[:] = np.nan

for ni in range(len(PARAM_GRID["power"])):
    for nj in range(len(PARAM_GRID["n_neighbours"])):
        for ind in range(nfolds):

            #print("inverse distance power: " + str(PARAM_GRID["power"][pow_is[n]]))
            #print("number of neighbours: " + str(PARAM_GRID["n_neighbours"][n_js[n]]))

            args = {"X0": cv_kc[ind],
                "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                "x1": cv_uc[ind],
                "power": PARAM_GRID["power"][pow_is[ni]],
                "n_neighbours": PARAM_GRID["n_neighbours"][n_js[nj]]}
            
            cv_pred = interp(**args)

            Cv_pred.append(cv_pred)

            cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

            nrmse = root_mean_squared_error(
                y_true = cv_true,
                y_pred=cv_pred
                ) / np.nanstd(cv_true)
            

            Nrmse[pow_is[ni], n_js[nj], ind] = nrmse

        score = np.nanmean(Nrmse[pow_is[ni], n_js[nj], :])
        NRMSE[pow_is[ni], n_js[nj]] = score
        
        if best_score > score:
            best_score = score
            best_params = (
            PARAM_GRID["power"][pow_is[ni]],
            PARAM_GRID["n_neighbours"][n_js[nj]]
            )

print(best_score)
print(best_params)

print(pow_is)
print(n_js)
sns.heatmap(
    NRMSE.T, 
    square=True, 
    vmin=0, 
    cmap = "gist_yarg",
    linewidth = 1,
    linecolor = 'w',
    ax=axs)


row_labels = []

data = pd.DataFrame(NRMSE.T, columns = PARAM_GRID["power"], index = PARAM_GRID["n_neighbours"])

col_label = np.where( data == np.nanmin(data) )[0]
row_label = np.where( data == np.nanmin(data) )[1]

i = row_label[0]
j = col_label[0]

print(i)
print(j)
axs.add_patch(
    FancyBboxPatch(
        (i, j), 1, 1,
        boxstyle="round,pad=0.0001",
        linewidth=2,
        edgecolor="red",
        facecolor="none", clip_on=False
    ))
    
plt.tight_layout()
plt.show()






#print(NRMSE)
'''
# compute everything else
select_pow = best_params[0]
select_n = best_params[1]

select_pow_i = np.where(PARAM_GRID["power"] == best_params[0])[0]
select_n_j = np.where(PARAM_GRID["n_neighbours"] == best_params[1])[0]

def indices_from_radial_nb(i, j, size=100):
    from math import sqrt
    radius = int( 0.5 * (sqrt(size) - 1))

    i_s = [ i + row_dist for row_dist in np.linspace(-radius, radius, 2*radius+1) ]
    j_s = [ j + col_dist for col_dist in np.linspace(-radius, radius, 2*radius+1) ]

    return i_s, j_s

eval_i_s, eval_j_s = indices_from_radial_nb(
    select_pow_i, 
    select_n_j, initial / 2)


for eval_i, eval_j in itertools.product(eval_i_s, eval_j_s):
    eval_i = int(eval_i)
    eval_j = int(eval_j)
    for ind in range(nfolds):

        #print("inverse distance power: " + str(PARAM_GRID["power"][pow_is[n]]))
        #print("number of neighbours: " + str(PARAM_GRID["n_neighbours"][n_js[n]]))

        args = {"X0": cv_kc[ind],
            "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
            "x1": cv_uc[ind],
            "power": PARAM_GRID["power"][eval_i],
            "n_neighbours": PARAM_GRID["n_neighbours"][eval_j]}
        
        cv_pred = interp(**args)

        Cv_pred.append(cv_pred)

        cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

        nrmse = root_mean_squared_error(
            y_true = cv_true,
            y_pred=cv_pred
            ) / np.nanstd(cv_true)
        

        Nrmse[eval_i, eval_j, ind] = nrmse

    score = np.nanmean(Nrmse[eval_i, eval_j, :])
    NRMSE[eval_i, eval_j] = score
    
    if best_score > score:
        best_score = score
        best_params = (
        PARAM_GRID["power"][eval_i],
        PARAM_GRID["n_neighbours"][eval_j]
        )

print(best_score)
print(best_params)
'''
'''
row_scores = np.zeros(len(PARAM_GRID["n_neighbours"]))

for nni, nn in enumerate(PARAM_GRID["n_neighbours"]):

    nrmses = np.zeros(nfolds)

    for ind in range(nfolds):
        args = {"X0": cv_kc[ind],
            "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
            "x1": cv_uc[ind],
            "power": select_pow,
            "n_neighbours": nn}
        
        cv_pred = interp(**args)

        Cv_pred.append(cv_pred)

        cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

        nrmse = root_mean_squared_error(
            y_true = cv_true,
            y_pred=cv_pred
            ) / np.nanstd(cv_true)
        

        nrmses[ind] = nrmse

    row_scores[nni] = np.nanmean(nrmses)

#print(row_scores)

column_scores = np.zeros(len(PARAM_GRID["power"]))

for pi, p in enumerate(PARAM_GRID["power"]):

    nrmses = np.zeros(nfolds)

    for ind in range(nfolds):
        args = {"X0": cv_kc[ind],
            "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
            "x1": cv_uc[ind],
            "power": p,
            "n_neighbours": select_n}
        
        cv_pred = interp(**args)

        Cv_pred.append(cv_pred)

        cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

        nrmse = root_mean_squared_error(
            y_true = cv_true,
            y_pred=cv_pred
            ) / np.nanstd(cv_true)
        

        nrmses[ind] = nrmse

    column_scores[pi] = np.nanmean(nrmses)

#print(column_scores)


#NRMSE[p] = Nrmse
'''
'''
df = pd.DataFrame.from_dict(
    NRMSE
    )

df_long = df.melt(
var_name = "power",
value_name="nrmse"
)
sns.lineplot(df_long, x="power", y="nrmse", color = percentage_colours[pi], errorbar='ci', ax=axs[ri])
axs[ri].set_title(f"range={ran}")
axs[ri].set_ylim([0,1.5])
# major ticks every 10
axs[ri].xaxis.set_major_locator(MultipleLocator(5))

# minor ticks at every point
axs[ri].xaxis.set_minor_locator(MultipleLocator(.1))
'''