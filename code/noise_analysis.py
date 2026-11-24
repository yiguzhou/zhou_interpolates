'''
Evaluate noise
'''

from config import *
import copy
from figs_config import *
from interpmodules import deterministic, helpers, stochastic
import itertools
import joblib
import matplotlib.pyplot as plt
from math import sqrt
import numpy as np
import pandas as pd
import pyvista as pv
import random
from run_wrappers import generate_covariates
from scipy.spatial import cKDTree
from scipy.stats import zscore
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

NOISE_LEVELS = [1, 5, 10, 15, 20, 50]

#--------helper functions--------#

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


#--------set up GRF plus noise--------#
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")

ranges = [25, 50, 75, 100]


#-------visualization--------#
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
ran=50
pct=5
verbose=1

PARAM_GRID = {
    "kernel": np.asarray([
                    "linear",
                    "thin_plate_spline",
                    "cubic",
                    "quintic",
                    "multiquadric",
                    "inverse_multiquadric",
                    "inverse_quadratic",
                    "gaussian"]),
    "n_neighbours": np.linspace(2, 20, 19),
    "power": np.linspace(2, 20, 73, dtype = np.float32)
}

sampl = joblib.load(projdir + f'/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{pct}.pkl')
kc, kv, uc, uv = sampl[1]

data = zscore(sphere_mesh[str(ran)])

sparse_coords = kc[:, :, 0]
sparse_data = np.asarray(data[kv.flatten().astype(int)])
test_coords = uc[:, :, 0]
test_data = np.asarray(data[uv.flatten().astype(int)])

nfolds = 10

cv_kc, cv_ki, cv_uc, cv_ui = get_cv_splits(sparse_coords, nfolds = nfolds, rseed=321)

SCORES = dict.fromkeys(approaches)
PREDS = dict.fromkeys(approaches)

idw=False
if idw:
    #--------idw--------#
    # cv-chosen power (grid search)
    if verbose: print("--------Starting IDW interpolation--------")

    interp = deterministic.interpolate_IDW

    Nrmse = {}

    for p in PARAM_GRID["power"]:

        nrmse = []
        cv_preds = []

        for ind in range(nfolds):

            args = {
                "X0": cv_kc[ind],
                "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                "x1": cv_uc[ind],
                "power": p
                }
            
            cv_pred = interp(**args)
            cv_preds.append(cv_pred)

            cv_true = np.asarray(sparse_data[cv_ui[ind].flatten().astype(int)])

            nrmse_cv = root_mean_squared_error(
                y_true = cv_true,
                y_pred=cv_pred
                ) / np.nanstd(cv_true)
            
            nrmse.append(nrmse_cv)

        Nrmse[p] = nrmse

    df = pd.DataFrame.from_dict(Nrmse)

    df_long = df.melt(
    var_name = "power",
    value_name="nrmse"
    )

    best_power = df_long["power"][
        df_long.sort_values(
            by="nrmse",
            ascending=True
            )["power"].index[0]
            ]

    if verbose: print(f"found best IDW power: {best_power:.2f}")
    update_args = {"X0": sparse_coords,
                    "Y0": sparse_data,
                    "x1": test_coords,
                    "power": best_power}

    y1 = interp(**update_args)
    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    SCORES["idw"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"IDW (power={best_power:.2f}) completed \n" +
        f"with NRMSE score {SCORES['idw']:.5f} (same as {((1 - (SCORES['idw'])**2) * 100):.2f}% variance explained)")

#--------knn--------#
knn=False
if knn:
    # cv-chosen power and neighbours (random then narrow search)
    if verbose: print("--------Starting KNN interpolation--------")

    PARAM_GRID["n_neighbours"] = np.linspace(2, 20, 19)

    interp = deterministic.interpolate_KNN

    initial = 10
    random.seed(321)
    pow_is = [ random.randint(0, len(PARAM_GRID["power"]) - 1) for _ in range(initial) ]
    n_js = [ random.randint(0, len(PARAM_GRID["n_neighbours"]) - 1) for _ in range(initial) ]

    best_score = 1
    best_params = (
        PARAM_GRID["power"][0],
        PARAM_GRID["n_neighbours"][0]
    )

    if verbose: print(f"Round 1: random search with initial draw of {initial} combinations")
    Nrmse = np.empty((len(PARAM_GRID["power"]), len(PARAM_GRID["n_neighbours"])))
    Nrmse[:] = np.nan
    nrmse = np.empty((len(PARAM_GRID["power"]), len(PARAM_GRID["n_neighbours"]), nfolds))
    nrmse[:] = np.nan

    for ni in pow_is:
        for nj in n_js:
            for ind in range(nfolds):

                args = {"X0": cv_kc[ind],
                    "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                    "x1": cv_uc[ind],
                    "power": PARAM_GRID["power"][ni],
                    "n_neighbours": PARAM_GRID["n_neighbours"][nj]}
                
                cv_pred = interp(**args)

                cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

                nrmse_cv = root_mean_squared_error(
                    y_true = cv_true,
                    y_pred=cv_pred
                    ) / np.nanstd(cv_true)
                
                nrmse[ni, nj, ind] = nrmse_cv

            score = np.nanmean(nrmse[ni, nj, :])
            Nrmse[ni, nj] = np.nanmean(nrmse[ni, nj, :])
            
            if best_score > score:
                best_score = score
                best_params = (
                PARAM_GRID["power"][ni],
                PARAM_GRID["n_neighbours"][nj]
                )

    closest_many = initial * 5
    if verbose: print(f"Round 2: search within {closest_many} closest combinations")
    select_pow = best_params[0]
    select_n = best_params[1]

    select_pow_i = np.where(PARAM_GRID["power"] == best_params[0])[0]
    select_n_j = np.where(PARAM_GRID["n_neighbours"] == best_params[1])[0]

    eval_i_s, eval_j_s = indices_from_radial_nb(
        select_pow_i, 
        select_n_j, closest_many)

    for eval_i, eval_j in itertools.product(eval_i_s, eval_j_s):
        eval_i = int(eval_i)
        eval_j = int(eval_j)

        for ind in range(nfolds):

            args = {"X0": cv_kc[ind],
                "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                "x1": cv_uc[ind],
                "power": PARAM_GRID["power"][eval_i],
                "n_neighbours": PARAM_GRID["n_neighbours"][eval_j]}
            
            cv_pred = interp(**args)

            cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

            nrmse_cv = root_mean_squared_error(
                y_true = cv_true,
                y_pred=cv_pred
                ) / np.nanstd(cv_true)
            
            nrmse[eval_i, eval_j, ind] = nrmse_cv

        score = np.nanmean(nrmse[eval_i, eval_j, :])
        Nrmse[eval_i, eval_j] = score
        
        if best_score > score:
            best_score = score
            best_params = (
            PARAM_GRID["power"][eval_i],
            PARAM_GRID["n_neighbours"][eval_j]
            )

    update_args = {
        "X0": sparse_coords,
        "Y0": sparse_data,
        "x1": test_coords,
        "power": best_params[0],
        "n_neighbours": best_params[1]
        }

    y1 = interp(**update_args)
    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    SCORES["knn"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"KNN (power={best_params[0]}, n_nbs={best_params[1]}) completed with \n" + 
        f"NRMSE score {SCORES['knn']:.5f} (same as {((1 - (SCORES['knn'])**2) * 100):.2f}% variance explained)"
        )

#--------rbf--------#
rbf=True
if rbf:
    if verbose: print("--------Starting RBF interpolation--------")

    # cv-chosen kernel and neighbours (grid search)
    interp = deterministic.interpolate_RBF

    initial = 10
    random.seed(321)
    kern_is = [ random.randint(0, len(PARAM_GRID["kernel"]) - 1) for _ in range(initial) ]
    n_js = [ random.randint(0, len(PARAM_GRID["n_neighbours"]) - 1) for _ in range(initial) ]

    best_score = 1
    best_params = (
        PARAM_GRID["kernel"][0],
        PARAM_GRID["n_neighbours"][0]
    )

    if verbose: print(f"Round 1: random search with initial draw of {initial} combinations")
    Nrmse = np.empty((len(PARAM_GRID["kernel"]), len(PARAM_GRID["n_neighbours"])))
    Nrmse[:] = np.nan
    nrmse = np.empty((len(PARAM_GRID["kernel"]), len(PARAM_GRID["n_neighbours"]), nfolds))
    nrmse[:] = np.nan

    for ni in kern_is:
        for nj in n_js:
            for ind in range(nfolds):

                args = {"X0": cv_kc[ind],
                    "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                    "x1": cv_uc[ind],
                    "kernel": PARAM_GRID["kernel"][ni],
                    "n_neighbours": PARAM_GRID["n_neighbours"][nj]}

                try:
                    cv_pred = interp(**args)
                    if verbose: print(cv_pred)
                    cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

                    nrmse_cv = root_mean_squared_error(
                        y_true = cv_true,
                        y_pred=cv_pred
                        ) / np.nanstd(cv_true)
                    
                    nrmse[ni, nj, ind] = nrmse_cv
                except ValueError:
                    
                    nrmse[ni, nj, ind] = np.nan

            score = np.nanmean(nrmse[ni, nj, :])
            Nrmse[ni, nj] = np.nanmean(nrmse[ni, nj, :])
            
            if best_score > score:
                best_score = score
                best_params = (
                PARAM_GRID["kernel"][ni],
                PARAM_GRID["n_neighbours"][nj]
                )

    closest_many = initial * 2
    if verbose: print(f"Round 2: search within {closest_many} closest combinations")
    select_pow = best_params[0]
    select_n = best_params[1]

    select_pow_i = np.where(PARAM_GRID["kernel"] == best_params[0])[0]
    select_n_j = np.where(PARAM_GRID["n_neighbours"] == best_params[1])[0]

    eval_i_s, eval_j_s = indices_from_radial_nb(
        select_pow_i, 
        select_n_j, closest_many)

    for eval_i, eval_j in itertools.product(eval_i_s, eval_j_s):
        eval_i = int(eval_i)
        eval_j = int(eval_j)

        for ind in range(nfolds):

            args = {"X0": cv_kc[ind],
                "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                "x1": cv_uc[ind],
                "kernel": PARAM_GRID["kernel"][eval_i],
                "n_neighbours": PARAM_GRID["n_neighbours"][eval_j]}
            
            cv_pred = interp(**args)

            cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

            nrmse_cv = root_mean_squared_error(
                y_true = cv_true,
                y_pred=cv_pred
                ) / np.nanstd(cv_true)
            
            nrmse[eval_i, eval_j, ind] = nrmse_cv

        score = np.nanmean(nrmse[eval_i, eval_j, :])
        Nrmse[eval_i, eval_j] = score
        
        if best_score > score:
            best_score = score
            best_params = (
            PARAM_GRID["kernel"][eval_i],
            PARAM_GRID["n_neighbours"][eval_j]
            )

    update_args = {
        "X0": sparse_coords,
        "Y0": sparse_data,
        "x1": test_coords,
        "kernel": best_params[0],
        "n_neighbours": best_params[1]
        }

    y1 = interp(**update_args)
    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    SCORES["rbf"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"RBF (power={best_params[0]}, n_nbs={best_params[1]}) completed with \n" + 
        f"NRMSE score {SCORES['rbf']:.5f} (same as {((1 - (SCORES['rbf'])**2) * 100):.2f}% variance explained)"
        )

    exhaustive=False
    if exhaustive:
        for i in range(len(PARAM_GRID["kernel"])):
            for j in range(len(PARAM_GRID["n_neighbours"])):
                for ind in range(nfolds):

                    args = {"X0": cv_kc[ind],
                        "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                        "x1": cv_uc[ind],
                        "kernel": PARAM_GRID["kernel"][i],
                        "n_neighbours": PARAM_GRID["n_neighbours"][j]}

                    try:
                        cv_pred = interp(**args)
                        cv_true = np.asarray(sparse_data[cv_ui[ind].flatten().astype(int)])

                        nrmse_cv = root_mean_squared_error(
                            y_true = cv_true,
                            y_pred=cv_pred
                            ) / np.nanstd(cv_true)
                        
                        nrmse[i, j, ind] = nrmse_cv
                    except ValueError:
                        nrmse[i, j, ind] = np.nan

                score = np.nanmean(nrmse[i, j, :])
                Nrmse[i, j] = score
                        
                if best_score > score:
                    best_score = score
                    best_params = (
                    PARAM_GRID["kernel"][i],
                    PARAM_GRID["n_neighbours"][j]
                    )

        update_args = {
            "X0": sparse_coords,
            "Y0": sparse_data,
            "x1": test_coords,
            "kernel": best_params[0],
            "n_neighbours": best_params[1]
            }

        y1 = interp(**update_args)
        full_map = np.zeros(sphere_mesh.n_points)
        full_map[kv.flatten().astype(int)] = sparse_data
        full_map[uv.flatten().astype(int)] = y1

        SCORES["rbf"] = root_mean_squared_error(
            y_true = test_data,
            y_pred = y1
            ) / np.nanstd(test_data)

        if verbose: print(
            f"RBF ({best_params[0]}, n_nbs={best_params[1]}) completed with \n" + 
            f"NRMSE score {SCORES['rbf']:.5f} (same as {((1 - (SCORES['rbf'])**2) * 100):.2f}% variance explained)"
            )

stochastic=False
if stochastic:
    #--------krige--------#
    # gaussian variogram - because we know that's how the data was made
    if verbose: print("--------Starting OK interpolation--------")

    y1, _ = stochastic.interpolate_Krige(
        X0=sparse_coords,
        Y0=sparse_data,
        x1=test_coords,
        model = 'gaussian',
        method='Ordinary')

    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    SCORES["ok"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"OK (exponential) completed with \n" + 
        f"NRMSE score {SCORES['ok']:.5f} (same as {((1 - (SCORES['ok'])**2) * 100):.2f}% variance explained)"
        )

    #--------regression kriging--------#
    if verbose: print("--------Starting RK interpolation--------")
    # choose 1-5 covariates
    Yk0, yk1 = generate_covariates(
        curr_map=ran,
        src_polydata=sphere_mesh,
        trg_polydata=sphere_mesh,
        nspins=1000, datasource='grf')

    y1, _ = stochastic.interpolate_Krige(
            X0=sparse_coords,
            Y0=sparse_data,
            x1=test_coords,
            Yk0=Yk0,
            yk1=yk1,
            model = 'gaussian',
            method='Regression')
    
    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    SCORES["rk"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"RK (gaussian) completed with \n" + 
        f"NRMSE score {SCORES['rk']:.5f} (same as {((1 - (SCORES['rk'])**2) * 100):.2f}% variance explained)"
        )

    #--------swr--------#
    # choose 1-5 covariates
    y1 = stochastic.interpolate_SWR(
        X0=sparse_coords,
        Y0=sparse_data,
        Yk0=Yk0,
        x1=test_coords,
        yk1=yk1
    )
    SCORES["swr"] = root_mean_squared_error(
            y_true = test_data,
            y_pred = y1
            ) / np.nanstd(test_data)
    
    if verbose: print(
        f"SWR completed with \n" + 
        f"NRMSE score {SCORES['swr']:.5f} (same as {((1 - (SCORES['swr'])**2) * 100):.2f}% variance explained)"
        )
    #--------compile scores--------#