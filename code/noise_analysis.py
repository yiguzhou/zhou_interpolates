'''
Evaluate noise
'''

# take noise level as the argument
import sys
if __name__ == "__main__":
    sigma = int(sys.argv[1])

from config import *
import copy
from figs_config import *
from interpmodules import deterministic, helpers, stochastic
import itertools
import joblib
import matplotlib.pyplot as plt
from math import sqrt
from noise_helpers import *
import numpy as np
import pandas as pd
import pyvista as pv
import random
from run_wrappers import generate_covariates
from scipy.spatial import cKDTree
from scipy.stats import zscore
import seaborn as sns
from sklearn.metrics import root_mean_squared_error


#--------set up GRF plus noise--------#
NOISE_LEVELS = [1, 5, 10, 15, 20, 50]

sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")

ranges = [25, 50, 75, 100]

plot=False
idw=True
knn=True
rbf=True
ordinary=True
regkrige=True
swr=True
run_stochastic=ordinary or regkrige or swr
need_covariates = regkrige or swr


#-------visualization--------#
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
no_noise = False
non_stationary = True
noisetype = "stationary"
if non_stationary: noisetype = "ns"


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

original = zscore(sphere_mesh[str(ran)])
noise = zscore(make_some_noise(sphere_mesh, sigma))
if no_noise:
    noise = np.zeros(sphere_mesh.n_points)
    sigma=0

if non_stationary:
    x_s = sphere_mesh.points[:, 0]
    sigma_i = linear_scaling(x_s, [-100, 100], [1, 20])
    ns = gaussian_gradual_blur(sphere_mesh, noise, sigma_i)
    noise = zscore(ns)

data = zscore(original + noise)

sparse_coords = kc[:, :, 0]
sparse_data = np.asarray(data[kv.flatten().astype(int)])
test_coords = uc[:, :, 0]
test_data = np.asarray(original[uv.flatten().astype(int)])

nfolds = 10

cv_kc, cv_ki, cv_uc, cv_ui = get_cv_splits(sparse_coords, nfolds = nfolds, rseed=321)

SCORES = dict.fromkeys(approaches)
PREDS = dict.fromkeys(approaches)

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

    PREDS["idw"] = full_map
    SCORES["idw"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"IDW (power={best_power:.2f}) completed \n" +
        f"with NRMSE score {SCORES['idw']:.5f} (same as {((1 - (SCORES['idw'])**2) * 100):.2f}% variance explained)")

#--------knn--------#
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
        eval_i = int(eval_i) % len(PARAM_GRID["power"])
        eval_j = int(eval_j) % len(PARAM_GRID["n_neighbours"])

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

    PREDS["knn"] = full_map

    SCORES["knn"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"KNN (power={best_params[0]}, n_nbs={best_params[1]}) completed with \n" + 
        f"NRMSE score {SCORES['knn']:.5f} (same as {((1 - (SCORES['knn'])**2) * 100):.2f}% variance explained)"
        )

#--------rbf--------#
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
        eval_i = int(eval_i) % len(PARAM_GRID["kernel"])
        eval_j = int(eval_j) % len(PARAM_GRID["n_neighbours"])

        for ind in range(nfolds):

            args = {"X0": cv_kc[ind],
                "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                "x1": cv_uc[ind],
                "kernel": PARAM_GRID["kernel"][eval_i],
                "n_neighbours": PARAM_GRID["n_neighbours"][eval_j]}
            try:
                cv_pred = interp(**args)
             
                cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

                nrmse_cv = root_mean_squared_error(
                    y_true = cv_true,
                    y_pred=cv_pred
                    ) / np.nanstd(cv_true)
                
                nrmse[eval_i, eval_j, ind] = nrmse_cv
            except ValueError:
                nrmse[eval_i, eval_j, ind] = np.nan

        score = np.nanmean(nrmse[eval_i, eval_j, :])
        Nrmse[eval_i, eval_j] = score
        
        if best_score > score:
            best_score = score
            best_params = (
            PARAM_GRID["kernel"][eval_i],
            PARAM_GRID["n_neighbours"][eval_j]
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

    PREDS["rbf"] = full_map
    SCORES["rbf"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"RBF ({best_params[0]}, n_nbs={best_params[1]}) completed with \n" + 
        f"NRMSE score {SCORES['rbf']:.5f} (same as {((1 - (SCORES['rbf'])**2) * 100):.2f}% variance explained)"
        )

if run_stochastic:
    Dist = helpers.internal_distance_matrix(sphere_mesh.points)
if ordinary:
    #--------krige--------#
    # gaussian variogram - because we know that's how the data was made
    if verbose: print("--------Starting OK interpolation--------")

    y1, _ = stochastic.interpolate_Krige(
        X0=sparse_coords,
        Y0=sparse_data,
        x1=test_coords,
        Dist=Dist,
        model = 'gaussian',
        method='Ordinary')

    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    PREDS["ok"] = full_map
    SCORES["ok"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    if verbose: print(
        f"OK (exponential) completed with \n" + 
        f"NRMSE score {SCORES['ok']:.5f} (same as {((1 - (SCORES['ok'])**2) * 100):.2f}% variance explained)"
        )

if need_covariates:
    Yk,_ = generate_covariates(
        curr_map=str(ran),
        src_polydata=sphere_mesh,
        trg_polydata=sphere_mesh,
        nspins=1000, datasource='grf')

if regkrige:
    #--------regression kriging--------#
    if verbose: print("--------Starting RK interpolation--------")
    # choose 1-5 covariates
    
    y1, _ = stochastic.interpolate_Krige(
            X0=sparse_coords,
            Y0=sparse_data,
            Dist=Dist,
            x1=test_coords,
            Yk0=Yk[:, kv.flatten().astype(int)],
            yk1=Yk[:, uv.flatten().astype(int)],
            model = 'gaussian',
            method='Regression')
    
    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    SCORES["rk"] = root_mean_squared_error(
        y_true = test_data,
        y_pred = y1
        ) / np.nanstd(test_data)

    PREDS["rk"] = full_map
    if verbose: print(
        f"RK (gaussian) completed with \n" + 
        f"NRMSE score {SCORES['rk']:.5f} (same as {((1 - (SCORES['rk'])**2) * 100):.2f}% variance explained)"
        )

if swr:
    #--------swr--------#
    if verbose: print("--------Starting SWR interpolation--------")
    # choose 1-5 covariates
    y1 = stochastic.interpolate_SWR(
        X0=sparse_coords,
        Y0=sparse_data,
        Yk0=Yk[:, kv.flatten().astype(int)],
        x1=test_coords,
        yk1=Yk[:, uv.flatten().astype(int)]
    )

    full_map = np.zeros(sphere_mesh.n_points)
    full_map[kv.flatten().astype(int)] = sparse_data
    full_map[uv.flatten().astype(int)] = y1

    PREDS["swr"] = full_map
    SCORES["swr"] = root_mean_squared_error(
            y_true = test_data,
            y_pred = y1
            ) / np.nanstd(test_data)
    
    if verbose: print(
        f"SWR completed with \n" + 
        f"NRMSE score {SCORES['swr']:.5f} (same as {((1 - (SCORES['swr'])**2) * 100):.2f}% variance explained)"
        )
#--------compile scores--------#
joblib.dump(SCORES, projdir + f"/results/rev_{ran}mm_{pct}pct_sigma{sigma}noise_{noisetype}_scores.pkl")
joblib.dump(PREDS, projdir + f"/results/rev_{ran}mm_{pct}pct_sigma{sigma}noise_{noisetype}_preds.pkl")