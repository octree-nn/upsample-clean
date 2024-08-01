import os
import open3d
import trimesh
import numpy as np


def sample_points(read_path, write_folder_gt, write_folder_input, file_name, 
                  npoints_gt, npoints_input, eval_folder=None, npoints_eval=None):
    mesh = trimesh.load(read_path, force='mesh')
    points_gt, _ = trimesh.sample.sample_surface(mesh, npoints_gt)
    points_gt = np.asarray(points_gt).astype(np.float32)

    mesh = open3d.io.read_triangle_mesh(read_path)
    points_input = mesh.sample_points_poisson_disk(npoints_input)
    points_input = np.asarray(points_input.points).astype(np.float32)

    if eval_folder is not None:
        points_eval = mesh.sample_points_poisson_disk(npoints_eval)
        points_eval = np.asarray(points_eval.points).astype(np.float32)
        np.savetxt(os.path.join(eval_folder, file_name[:-3] + 'xyz'), points_eval)

    else:
        # normalize the points
        bbmin, bbmax = np.min(points_gt, axis=0), np.max(points_gt, axis=0)
        center = (bbmin + bbmax) / 2.0
        radius = 2.0 / (np.max(bbmax - bbmin) + 1.0e-6)
        points_gt = 0.5 * (points_gt - center) * radius  # normalize to [-0.5, 0.5]
        points_input = 0.5 * (points_input - center) * radius  # normalize to [-0.5, 0.5]

    np.save(os.path.join(write_folder_gt, file_name), points_gt)
    np.save(os.path.join(write_folder_input, file_name), points_input)
    