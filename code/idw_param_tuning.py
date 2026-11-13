'''
IDW hyperparameter tuning
'''

from config import *
from interpmodules import deterministic
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
from scipy.stats import zscore
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

PARAM_GRID = {
    'power': np.linspace(0.1, 20, 200, dtype = np.float32)
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


#---------set up--------#
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")
interp = deterministic.interpolate_IDW

#ran=50
ranges = [25, 100]
#pct=5
percentages = [5, 10, 25, 50]

cmap_cont = plt.cm.Reds
percentage_colours = cmap_cont(np.linspace(0.25,1,len(percentages)))


fig, axs = plt.subplots(2, 1, figsize = (30, 9), sharex=True, sharey=True)

for ri, ran in enumerate(ranges):
    for pi, pct in enumerate(percentages):
        sampl = joblib.load(projdir + f"/data/sampling/fsLR-4k_sphere_samp-fibonacci_pct-{pct}.pkl")
        kc, kv, uc, uv = sampl[1]
        data = zscore(sphere_mesh[str(ran)])

        sparse_coords = kc[:, :, 0]
        sparse_data = data[kv.flatten().astype(int)]
        test_coords = uc[:, :, 0]


        nfolds = 10
        Cv_pred = []


        cv_kc, cv_ki, cv_uc, cv_ui = get_cv_splits(sparse_coords, nfolds = 20, rseed=321)

        NRMSE = {}

        for p in PARAM_GRID["power"]:

            Nrmse = []

            for ind in range(nfolds):

                args = {"X0": cv_kc[ind],
                    "Y0": sparse_data[cv_ki[ind].flatten().astype(int)],
                    "x1": cv_uc[ind],
                    "power": p}
                
                cv_pred = interp(**args)

                Cv_pred.append(cv_pred)

                cv_true = sparse_data[cv_ui[ind].flatten().astype(int)]

                nrmse = root_mean_squared_error(
                    y_true = cv_true,
                    y_pred=cv_pred
                    ) / np.nanstd(cv_true)
                Nrmse.append(nrmse)

            NRMSE[p] = Nrmse

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

plt.tight_layout(); plt.savefig(projdir + "/figs/rev_idw_grf_finetune.svg")
#import matplotlib.pyplot as plt

#plt.figure(figsize = (3,3))
#plt.boxplot(NRMSE)
#plt.show()
#--------test--------#