'''
plot clean gaussian grf
'''

from config import *
from figs_config import *
import gstools as gs
from interpmodules import helpers
import joblib
import matplotlib.pyplot as plt
import math
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import root_mean_squared_error

#--------Set up--------#
PV = [25, 50, 75, 100] # percentage of the max distance as the max of the variogram x-axis
NH = np.linspace(10, 100, 15) # number of bins, enclosing 1% to 10% of max distance per bin

ranges = [10, 25, 50, 75, 100]
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")
nreps = 100

Dist = helpers.internal_distance_matrix(v = sphere_mesh.points, method = "euclidean")

#--------Theoretical variograms--------#
cmap_cont = plt.cm.Blues
range_colours = cmap_cont(np.linspace(0.25, 1, len(ranges)))

plt.figure(figsize = (4,2))

for ri, ran in enumerate(ranges):

    x = np.linspace(0, Dist.max())
    l = helpers.convert_to_length(r=ran, model="gaussian")
    theoretical = gs.covmodel.Gaussian(dim=3, len_scale=l, var=1).variogram(x)
    plt.plot(x, theoretical, color = range_colours[ri], linewidth = 2, zorder = 10)
    plt.axvline(ran, color = range_colours[ri], linewidth = 1)

plt.ylim([0, 1.2])
plt.tight_layout()
plt.savefig(projdir + "/figs/rev_gaussian_theoretical.svg")

#--------Binwidths and truncations--------#

for ri, ran in enumerate(ranges):

    Data = []
    
    instances = [ helpers.make_random_maps(ran=ran, mesh=sphere_mesh, seed=i) for i in range(nreps) ]
    
    for pvi in PV:
        H = {}
        EV = {}
        NCOUNTS = {}
        for nhi in NH:
            for ii in range(nreps):
        
                nhi = int(nhi)
                
                h, ev, N = helpers.get_empirical_variogram_fast(x=instances[ii],X=sphere_mesh.points,
                                                                D=Dist,nh=int(nhi),max_lag=np.nanmax(Dist)*pvi/100,
                                                                h_bounds='auto', method='euclidean')
                if ii == 0:
                    H[nhi] = [h]
                    EV[nhi] = [ev]
                else: 
                    H[nhi].append(h)
                    EV[nhi].append(ev)
                    
                NCOUNTS[nhi] = N
    
        Scores = np.zeros((len(NH), nreps))
        
        for ii in range(nreps):
            scores = np.ones(len(NH))
            
            for ind, nhi in enumerate(NH):
        
                nhi = int(nhi)
                
                x = H[nhi][ii]
                
                theoretical = [ helpers.gaussian(h=xi, r=ran, c0=1, nugget=0) for xi in x ]
            
                if np.isnan(EV[nhi]).any():
                    EV[nhi][ii][np.where(np.isnan(EV[nhi][ii]))[0]] = 0
                        
                scores[ind] = root_mean_squared_error(theoretical, EV[nhi][ii]) / np.std(theoretical)
                
            Scores[:,ii] = scores
        
        Scores = pd.DataFrame(Scores.T)
        Scores.columns = NH
    
        Scores_long = Scores.melt(
        var_name='nh',
        value_name='error'
        )
        
        Scores_long['nh'] = Scores_long['nh'].astype(int)
        Scores_long['nh'] = Scores_long['nh'].astype(int)
        Scores_long['pv'] = pvi
        
        Data.append(Scores_long)
            
        joblib.dump((ran, Data), projdir + f"/results/rev_binwidth_simulations_{ran}mm_100reps.pkl")


fig, axs = plt.subplots(1, len(ranges), figsize = (2*len(ranges) + 2 ,2), sharey=True, sharex=True)

for ri, ran in enumerate(ranges):

    ran, Data = joblib.load(projdir + f"/results/rev_binwidth_simulations_{ran}mm_100reps.pkl")

    df = pd.concat(Data)
    
    sns.lineplot(df, x='nh', y='error', hue ='pv', errorbar='ci', ax=axs[ri], legend=None)

    
    
    axs[ri].axhline(math.sqrt(.1), c= 'k', linestyle = 'dotted')
    axs[ri].axhline(math.sqrt(.05), c= 'k', linestyle = 'dotted')


    if ri == len(ranges) - 1:
        sns.lineplot(df, x='nh', y='error', hue ='pv', errorbar='ci', ax=axs[ri])
        sns.move_legend(axs[len(ranges)-1], "upper left", bbox_to_anchor=(1, 1))
    
plt.ylim([0,1.2])
plt.ylabel("NRMSE")
#extraticks = [math.sqrt(0.1), math.sqrt(0.05)]
#plt.yticks(list(plt.yticks()[0]) + extraticks)
plt.tight_layout()
plt.savefig(projdir + '/figs/rev_binwidth_simulations_100reps.svg')