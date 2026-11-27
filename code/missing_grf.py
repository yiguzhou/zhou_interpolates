'''
Data missingness - GRF
'''

from config import *
from figs_config import *
import numpy as np
from scipy.spatial import SphericalVoronoi, KDTree
from interpmodules import helpers
import joblib
import pyvista as pv
import random

#--------helper functions--------#
def _find_duplicates_across_arrays(list_of_arrays):

    all_dups = []
    
    for i in range(len(list_of_arrays)):
        for j in range(len(list_of_arrays)):
            if i == j: continue
            all_dups += (np.intersect1d(list_of_arrays[i], list_of_arrays[j]).tolist())
    
    return all_dups

def gaussian_sphere(mesh_vertices, center, sigma):

    # compute gaussian weights over the spherical mesh centered around a focal vertex
    
    center = center / np.linalg.norm(center)
    mesh_vertices = mesh_vertices / np.linalg.norm(mesh_vertices, axis=1, keepdims=True) # normalize
    dot_products = np.clip(np.dot(mesh_vertices, center), -1.0, 1.0)
    angles = np.arccos(dot_products)  # Angular distances in radians

    values = np.exp(-(angles**2) / (2 * sigma**2))
    return values

def _coords_icosahedron_to_perfect_sphere(vertices, radius=1.0):
    norms = np.linalg.norm(vertices, axis=1, keepdims=True)
    return radius * vertices / norms

def slerp(p0, p1, num=10):
    """Spherical linear interpolation between two unit vectors."""
    p0 = p0 / np.linalg.norm(p0)
    p1 = p1 / np.linalg.norm(p1)
    omega = np.arccos(np.clip(np.dot(p0, p1), -1.0, 1.0))
    if np.isclose(omega, 0):
        return np.tile(p0, (num, 1))
    sin_omega = np.sin(omega)
    t = np.linspace(0, 1, num)[:, None]
    return (np.sin((1 - t) * omega) / sin_omega) * p0 + (np.sin(t * omega) / sin_omega) * p1


#--------setting up--------#
sphere_mesh = joblib.load(projdir + "/data/fsLR-4k_sphere_grf.pkl")
focal = False
focal_diffuse = True
plot = True
verbose = True

#--------focal sampling--------#
D = helpers.internal_distance_matrix(sphere_mesh.points)
tree = KDTree(sphere_mesh.points)
max_lag = np.nanmax(D)
spdist = tree.sparse_distance_matrix(tree, max_distance = max_lag, output_type="coo_matrix")

pct = 5
n_foci = np.linspace(1, 10, 10, dtype = int)
seed = random.seed(321)

n_goal = int(sphere_mesh.n_points * pct/100)


if focal:
    focal_samples = dict.fromkeys(n_foci)

    centroids = []

    for n in n_foci:
        # centroids for the foci
        centroids.append(random.randint(0, sphere_mesh.n_points))

        centroids_coords = [ sphere_mesh.points[centroid] for centroid in centroids ]


        neighbours = dict.fromkeys(centroids)
        
        # evenly distribute the neighbours among centroid points
        # overlap neighbourhoods

        n_neighbours = n_goal // n

        for i, centroid_coords in enumerate(centroids_coords):
            _, neighbours[centroids[i]] = tree.query(x = centroid_coords, k = n_neighbours)

        overlaps = _find_duplicates_across_arrays([neighbours[centroid] for centroid in centroids])
        n_overlaps = len(overlaps)

        fv = np.concatenate([neighbours[centroid] for centroid in centroids], dtype = int)
        fv = np.unique(fv, return_counts = False)

        while n_overlaps > 0:

            add = n_neighbours + n_overlaps // n

            extended_nbs = []

            for i, centroid_coords in enumerate(centroids_coords):
                _, extended_neighbours = tree.query(x = centroid_coords, k = add)
                extended_nbs.append(extended_neighbours[add:])
            
            extended_nbs = np.concat(extended_nbs)
            fv_add = np.unique(extended_nbs, return_counts = False)

            overlaps = _find_duplicates_across_arrays([fv, fv_add])
            n_overlaps = len(overlaps)
            
            fv = np.unique(np.concat([fv, fv_add]))
            if fv.shape[0] == sphere_mesh.n_points: break
            #if verbose: print(n_overlaps)


        fv = fv
        fc = sphere_mesh.points[fv]

        focal_samples[n] = fv, fc


    #====voronoi area====#
    perfect_sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)

    if plot:
        pl = pv.Plotter(shape = (1, len(n_foci)), window_size = (400 * len(n_foci), 400))

        for fi, n in enumerate(n_foci):

            fv, fc = focal_samples[n]
            perfect_points = _coords_icosahedron_to_perfect_sphere(vertices=fc, radius=1)
            #sv = SphericalVoronoi(perfect_points)
            #sv.sort_vertices_of_regions()

            pl.subplot(0, fi)
            pl.add_points(
            perfect_points,
            point_size=10,
            scalars = sphere_mesh["50"][fv],
            cmap = bpr,
            **point_aes)

            pl.add_mesh(perfect_sphere, color="k", opacity=0.7, smooth_shading=True)
            pl.add_text(f"{str(n)} foci", font_size = 10, font = "courier", position = "lower_edge")
            # Add Voronoi edges as spherical arcs
            #for region in sv.regions:
            #    verts = sv.vertices[region]
            #    for i in range(len(verts)):
            #        j = (i + 1) % len(verts)
            #        arc = slerp(verts[i], verts[j], num=20)
            #        line = pv.lines_from_points(arc)
            #        pl.add_mesh(line, color="white", line_width=2)

            pl.camera_position="yz"
        pl.show(jupyter_backend='static')


#====spatial sampling rate====#


#--------focal-diffuse sampling--------#
if focal_diffuse:

    focal_diffuse_samples = dict.fromkeys(n_foci)

    centroids = []
    centroids_coords = []

    n_foci = [1, 2, 3]
    for n in n_foci:
        # evenly distribute the neighbours among centroid points
        # overlap neighbourhoods
        # centroids for the foci

        centroid = random.randint(0, sphere_mesh.n_points)
        centroids.append(centroid)

        centroids_coords.append(sphere_mesh.points[centroid])

        neighbours = dict.fromkeys(centroids)
        weights = dict.fromkeys(centroids)
        probabilities = dict.fromkeys(centroids)
        n_neighbours = n_goal // n
        print(centroids_coords)


        for i in range(n):
            _, nbs = tree.query(x=centroids_coords[i], k=sphere_mesh.n_points - 1)
            neighbours[centroids[i]] = nbs
            weights[centroids[i]] = gaussian_sphere(sphere_mesh.points, centroids_coords[i], .5)
            probabilities[centroids[i]] = weights[centroids[i]] / np.sum(weights[centroids[i]])

        new_array = np.zeros(n_goal)

        n_combined = 0

        sample_count = n_goal // n

        while n_combined != n_goal:

            if sample_count > sphere_mesh.n_points: 
                centroids[n - 1] = random.randint(0, sphere_mesh.n_points) # sample another one
                sample_count = n_goal // n
                print(f"replace centroid # {n}")

                

            sampled_indices = [ np.random.choice(sphere_mesh.n_points, size=sample_count, \
                                                replace=False, p=probabilities[centroids[i]]) for i in range(n) ]

            combined = np.unique(np.concatenate(sampled_indices, 0))
            n_combined = combined.shape[0]

            new_array = combined
            sample_count += 1

        fv = new_array
        fc = sphere_mesh.points[new_array.astype(int)]
        print(fv.shape)
        focal_diffuse_samples[n] = fv, fc

    #====voronoi area====#
    perfect_sphere = pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=60)

    if plot:
        pv.global_theme.allow_empty_mesh = True
        pl = pv.Plotter(shape = (1, len(n_foci)), window_size = (400 * len(n_foci), 400))

        for fi, n in enumerate(n_foci):

            fv, fc = focal_diffuse_samples[n]
            perfect_points = _coords_icosahedron_to_perfect_sphere(vertices=fc, radius=1)
            #sv = SphericalVoronoi(perfect_points)
            #sv.sort_vertices_of_regions()

            pl.subplot(0, fi)
            pl.add_points(
            perfect_points,
            point_size=10,
            scalars = sphere_mesh["50"][fv],
            cmap = bpr,
            **point_aes)

            pl.add_mesh(perfect_sphere, color="k", opacity=0.7, smooth_shading=True)
            pl.add_text(f"{str(n)} foci", font_size = 10, font = "courier", position = "lower_edge")
            # Add Voronoi edges as spherical arcs
            #for region in sv.regions:
            #    verts = sv.vertices[region]
            #    for i in range(len(verts)):
            #        j = (i + 1) % len(verts)
            #        arc = slerp(verts[i], verts[j], num=20)
            #        line = pv.lines_from_points(arc)
            #        pl.add_mesh(line, color="white", line_width=2)

            pl.camera_position="yz"
        pl.show(jupyter_backend='static')