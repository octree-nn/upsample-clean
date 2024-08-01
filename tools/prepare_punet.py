import os
import numpy as np
import open3d
import trimesh
from tqdm import tqdm
from normalize_mesh import normalize_off


def sample_training_points(read_path, write_folder_gt, write_folder_input, file_name):
    mesh = trimesh.load(read_path, force='mesh')
    points_gt, _ = trimesh.sample.sample_surface(mesh, 300000)
    points_gt = np.asarray(points_gt).astype(np.float32)

    mesh = open3d.io.read_triangle_mesh(read_path)
    points_input_50k = mesh.sample_points_poisson_disk(50000)
    points_input_50k = np.asarray(points_input_50k.points).astype(np.float32)

    points_input_10k = mesh.sample_points_poisson_disk(10000)
    points_input_10k = np.asarray(points_input_10k.points).astype(np.float32)

    # normalize the points
    bbmin, bbmax = np.min(points_gt, axis=0), np.max(points_gt, axis=0)
    center = (bbmin + bbmax) / 2.0
    radius = 2.0 / (np.max(bbmax - bbmin) + 1.0e-6)
    points_gt = 0.5 * (points_gt - center) * radius  # normalize to [-0.5, 0.5]
    points_input_50k = 0.5 * (points_input_50k - center) * radius  # normalize to [-0.5, 0.5]
    points_input_10k = 0.5 * (points_input_10k - center) * radius  # normalize to [-0.5, 0.5]

    np.save(os.path.join(write_folder_gt, file_name), points_gt)
    save_path = os.path.join(write_folder_input, '50k', file_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, points_input_50k)
    save_path = os.path.join(write_folder_input, '10k', file_name)
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.save(save_path, points_input_10k)

def sample_test_eval_points(read_mesh_path, read_points_path, write_folder_gt, write_folder_eval, write_folder_input, file_name):
    mesh = trimesh.load(read_mesh_path, force='mesh')
    points_gt, _ = trimesh.sample.sample_surface(mesh, 300000)
    points_gt = np.asarray(points_gt).astype(np.float32)

    mesh = open3d.io.read_triangle_mesh(read_mesh_path)
    points_eval_50k = mesh.sample_points_poisson_disk(50000)
    points_eval_50k = np.asarray(points_eval_50k.points).astype(np.float32)

    points_eval_10k = mesh.sample_points_poisson_disk(10000)
    points_eval_10k = np.asarray(points_eval_10k.points).astype(np.float32)

    # normalize the points
    bbmin, bbmax = np.min(points_gt, axis=0), np.max(points_gt, axis=0)
    center = (bbmin + bbmax) / 2.0
    radius = 2.0 / (np.max(bbmax - bbmin) + 1.0e-6)
    points_gt = 0.5 * (points_gt - center) * radius   

    points_eval_50k = 0.5 * (points_eval_50k - center) * radius
    points_eval_10k = 0.5 * (points_eval_10k - center) * radius  

    np.save(os.path.join(write_folder_gt, file_name), points_gt)
    save_path = os.path.join(write_folder_eval, 'PUNet_50k/gt_poisson', file_name[:-3]+'xyz')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.savetxt(save_path, points_eval_50k)
    save_path = os.path.join(write_folder_eval, 'PUNet_10k/gt_poisson', file_name[:-3]+'xyz')
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    np.savetxt(save_path, points_eval_10k)

    for res in ['1', '5']:
        for noise in ['1', '2', '25']:
            data = np.loadtxt(os.path.join(read_points_path, 'PUNet_%s0000_poisson_0.0%s' % (res, noise), file_name[:-3]+'xyz'))
            data = 0.5 * (data - center) * radius  
            save_path = os.path.join(write_folder_input, '%s0k/noise_%s' % (res, noise), file_name)
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            np.save(save_path, data)


raw_data_path = 'original_dataset/data_and_ckpt/data/PUNet/meshes/%s' # subset
save_training_path = 'data/train/%s/cleaning'
save_test_path = 'data/test/%s/cleaning'
save_eval_path = 'data/evaluate'

print('Sampling PUNet training dataset......')
subset_root = raw_data_path % 'train'
files = os.listdir(subset_root)
for file in tqdm(files, ncols=60):
    write_folder_gt = save_training_path % 'gt'
    if not os.path.exists(write_folder_gt):
        os.makedirs(write_folder_gt)
    write_folder_input = save_training_path % 'input'
    if not os.path.exists(write_folder_input):
        os.makedirs(write_folder_input)

    read_path = os.path.join(subset_root, file)
    name = file[:-3] + 'npy'
    sample_training_points(read_path, write_folder_gt, write_folder_input, name)

print('Normalize PUNet test meshes......')
evaluate_str = ''
subset_root = raw_data_path % 'test'
files = os.listdir(subset_root)
for file in tqdm(files, ncols=60):
    write_folder = os.path.join(save_eval_path, 'PUNet_50k/normalized_meshes')
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    save_name = file
    evaluate_str += save_name[:-3] + 'npy\n'
    read_path = os.path.join(subset_root, file)
    normalize_off(read_path, write_folder, save_name)

    write_folder = os.path.join(save_eval_path, 'PUNet_10k/normalized_meshes')
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    normalize_off(read_path, write_folder, save_name)

with open(os.path.join(save_eval_path, 'PUNet_50k/evaluatelist.txt'), 'w') as f:
    f.write(evaluate_str)
with open(os.path.join(save_eval_path, 'PUNet_10k/evaluatelist.txt'), 'w') as f:
    f.write(evaluate_str)

print('Sampling PUNet test and eval dataset......')
subset_root = raw_data_path % 'test'
files = os.listdir(subset_root)
for file in tqdm(files, ncols=60):
    write_folder_gt = save_test_path % 'gt'
    if not os.path.exists(write_folder_gt):
        os.makedirs(write_folder_gt)
    write_folder_eval = save_eval_path
    if not os.path.exists(write_folder_eval):
        os.makedirs(write_folder_eval)
    write_folder_input = save_test_path % 'input'
    if not os.path.exists(write_folder_input):
        os.makedirs(write_folder_input)

    read_mesh_path = os.path.join(subset_root, file)
    read_points_path = 'original_dataset/data_and_ckpt/data/examples'
    name = file[:-3] + 'npy'
    sample_test_eval_points(read_mesh_path, read_points_path, write_folder_gt, write_folder_eval, write_folder_input, name)
