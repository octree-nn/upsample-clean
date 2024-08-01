import numpy as np
import os
from tqdm import tqdm
from .utils import chamfer_distance as cd, hausdorff_distance as hd, p2f


def evaluate(metric, file_list, outputdir, targetdir):

    compute = eval(metric)

    test_data_list = file_list
    with open(test_data_list) as fid:
        filenames = fid.read()
    filenames = filenames.split()  
    length = len(filenames)
    print('length: ', length)

    total = 0.0
    pbar = tqdm(filenames, ncols=80)
    for name in pbar:
        out_dir = os.path.join(outputdir, name[:-4] +'.out.xyz')
        target_dir = os.path.join(targetdir, name[:-4] + '.xyz')
        total += compute(target_dir, out_dir)

    average = total / length
    print(f'average_%s: %.3e' % (metric, average))
    msg = f'average_%s: %.3e\n' % (metric, average)

    log_file = os.path.join(os.path.dirname(outputdir), 'evaluate.csv')
    with open(log_file, 'a') as fid:
        fid.write(msg)
