'''
visualize noise analysis
'''

from config import *
import copy
from figs_config import *
import joblib
from noise_helpers import *
import pyvista as pv
from scipy.stats import zscore


sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")

ran = 50
pct = 5

original = zscore(np.asarray(sphere_mesh[str(ran)])).flatten()

pl = pv.Plotter(
    shape = (len(NOISE_LEVELS) + 1, len(approaches) + 3),
    window_size = (400 * (len(approaches) + 3), 400 * len(NOISE_LEVELS))
    )

# the original
pl.subplot(0, 0)
tmp_mesh = copy.deepcopy(sphere_mesh)
pl.add_mesh(tmp_mesh, scalars = original, cmap = bpr, **mesh_aes)
pl.add_text(f"GRF({ran}mm)", font = "courier", font_size = 10, position = "lower_edge")
pl.camera_position = "yz"

scores = joblib.load(projdir + f"/results/rev_{ran}mm_{pct}pct_sigma0noise_scores.pkl")
preds = joblib.load(projdir + f"/results/rev_{ran}mm_{pct}pct_sigma0noise_preds.pkl")

for j in range(len(approaches)):
    pl.subplot(0, j + 3)
    tmp_mesh = copy.deepcopy(sphere_mesh)
    pl.add_mesh(tmp_mesh, scalars = preds[approaches[j]], cmap = bpr, **mesh_aes)
    pl.add_text(f"NRMSE={scores[approaches[j]]:.4f} \n" + 
                f"{((1 - scores[approaches[j]]**2)*100):.2f}% var expl",
                font = "courier", font_size = 10, position = "upper_left")

    pl.add_text(approaches[j], font = "courier", font_size = 10, position = "lower_edge")
    pl.camera_position = "yz"
    
# the noise
for i in range(len(NOISE_LEVELS)):

    sigma = NOISE_LEVELS[i]

    noise = zscore(np.asarray(make_some_noise(sphere_mesh, sigma))).flatten()
    data = zscore(original + noise)

    scores = joblib.load(projdir + f"/results/rev_{ran}mm_{pct}pct_sigma{sigma}noise_scores.pkl")
    preds = joblib.load(projdir + f"/results/rev_{ran}mm_{pct}pct_sigma{sigma}noise_preds.pkl")

    pl.subplot(i + 1, 1)
    tmp_mesh = copy.deepcopy(sphere_mesh)
    pl.add_mesh(tmp_mesh, scalars = noise, cmap = "gist_yarg", **mesh_aes)
    pl.add_text(f"+sigma={sigma}", font = "courier", font_size = 10, position = "lower_edge")
    pl.camera_position = "yz"

    # noisy data
    pl.subplot(i + 1, 2)
    tmp_mesh = copy.deepcopy(sphere_mesh)
    pl.add_mesh(tmp_mesh, scalars = data, cmap = bpr, **mesh_aes)
    pl.add_text("noisy", font = "courier", font_size = 10, position = "lower_edge")
    pl.camera_position = "yz"

    for j in range(len(approaches)):
        pl.subplot(i + 1, j + 3)
        tmp_mesh = copy.deepcopy(sphere_mesh)
        pl.add_mesh(tmp_mesh, scalars = preds[approaches[j]], cmap = bpr, **mesh_aes)
        pl.add_text(f"NRMSE={scores[approaches[j]]:.4f} \n" + 
                    f"{((1 - scores[approaches[j]]**2)*100):.2f}% var expl",
                    font = "courier", font_size = 10, position = "upper_left")

        pl.add_text(approaches[j], font = "courier", font_size = 10, position = "lower_edge")
        pl.camera_position = "yz"

#pl.show(jupyter_backend = "static")
pl.save_graphic(projdir + "/figs/rev_noisy_interpolation_examples.svg")
