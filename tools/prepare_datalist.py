import os
import numpy as np


test_root_path = 'data/test/input'
subsets = ['cleaning', 'upsampling']

teststr = ''

for subset in subsets:
    subset_path = os.path.join(test_root_path, subset)  # upsampling or cleaning
    subsubsets = os.listdir(subset_path)    
    for subsubset in subsubsets:
        subsubset_path = os.path.join(subset_path, subsubset)  # 10k or 50k or PU-GAN or PU1K
        if 'input/upsampling' in subsubset_path:
            files = os.listdir(subsubset_path)  # names of files
            for file in files:
                if file.endswith('xyz'):
                    continue
                teststr += '%s/%s/%s\n' % (subset, subsubset, file) # files
        else:
            final_folders = os.listdir(subsubset_path)  # names of noise_folders
            for final_folder in final_folders:
                files_path = os.path.join(subsubset_path, final_folder)  # noise_1
                files = os.listdir(files_path)
                for file in files:
                    if file.endswith('xyz'):
                        continue
                    teststr += '%s/%s/%s/%s\n' % (subset, subsubset, final_folder, file) # files

with open('data/testlist.txt', 'w') as f:
    f.write(teststr)


train_root_path = 'data/train/input'
train_str = ''

for subset in subsets:
    subset_path = os.path.join(train_root_path, subset)  # upsampling or cleaning
    subsubsets = os.listdir(subset_path)    
    for subsubset in subsubsets:
        subsubset_path = os.path.join(subset_path, subsubset)  # 10k or 50k or PU-GAN or PU1K
        files = os.listdir(subsubset_path)  # names of files
        if 'input/upsampling' in subsubset_path:
            for file in files:
                if file.endswith('xyz'):
                    continue
                train_str += '%s/%s/%s\n' % (subset, subsubset, file) # files
        else:
            for file in files:
                if file.endswith('xyz'):
                    continue
                train_str += '%s/%s/%s\n' % (subset, subsubset, file) # files
                train_str += '%s/%s/%s\n' % (subset, subsubset, file) # files
                train_str += '%s/%s/%s\n' % (subset, subsubset, file) # files

with open('data/trainlist.txt', 'w') as f:
    f.write(train_str)
