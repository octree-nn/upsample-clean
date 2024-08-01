import os
from tqdm import tqdm
from normalize_mesh import normalize_off
from sample_points import sample_points


save_training_path = 'data/train/%s/upsampling/Sketchfab'
save_test_path = 'data/test/%s/upsampling/Sketchfab'
save_eval_path = 'data/evaluate/Sketchfab'

print('Sampling Sketchfab training dataset......')
raw_training_path = 'original_dataset/mesh/train_mesh'

files = os.listdir(raw_training_path)
for file in tqdm(files, ncols=60):
    write_folder_gt = save_training_path % 'gt'
    if not os.path.exists(write_folder_gt):
        os.makedirs(write_folder_gt)
    write_folder_input = save_training_path % 'input'
    if not os.path.exists(write_folder_input):
        os.makedirs(write_folder_input)

    read_path = os.path.join(raw_training_path, file)
    name = "".join(filter(str.isalnum, file))
    name = name[:-3] + '.npy'
    sample_points(read_path, write_folder_gt, write_folder_input, name, 500000, 5000)

print('Normalize Sketchfab test meshes......')
evaluate_str = ''
subset_root = 'original_dataset/mesh/test_mesh'
files = os.listdir(subset_root)
for file in tqdm(files, ncols=60):
    write_folder = os.path.join(save_eval_path, 'normalized_meshes')
    if not os.path.exists(write_folder):
        os.makedirs(write_folder)
    save_name = "".join(filter(str.isalnum, file))
    save_name = save_name[:-3] + '.npy'
    evaluate_str += save_name + '\n'
    read_path = os.path.join(subset_root, file)
    normalize_off(read_path, write_folder, save_name[:-3]+'off')

with open(os.path.join(save_eval_path, 'evaluatelist.txt'), 'w') as f:
    f.write(evaluate_str)

print('Sampling Sketchfab test and eval dataset......')
normalized_mesh_folder = os.path.join(save_eval_path, 'normalized_meshes')
files = os.listdir(normalized_mesh_folder)
for file in tqdm(files, ncols=60):
    # 创建文件夹
    write_folder_gt = save_test_path % 'gt'
    if not os.path.exists(write_folder_gt):
        os.makedirs(write_folder_gt)
    write_folder_input = save_test_path % 'input'
    if not os.path.exists(write_folder_input):
        os.makedirs(write_folder_input)
    write_folder_eval = os.path.join(save_eval_path, 'gt_poisson')
    if not os.path.exists(write_folder_eval):
        os.makedirs(write_folder_eval)

    # 读取mesh
    read_path = os.path.join(normalized_mesh_folder, file)
    # 只保留文件名中的字母和数字，并将.off 后缀改为 .npy
    name = "".join(filter(str.isalnum, file))
    name = name[:-3] + '.npy'
    sample_points(read_path, write_folder_gt, write_folder_input, name, 
                  500000, 5000, write_folder_eval, 80000)
    