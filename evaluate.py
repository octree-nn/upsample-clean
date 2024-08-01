import argparse
from metrics import evaluate


parse = argparse.ArgumentParser(description='evaluate')
parse.add_argument('--outputdir', type=str)
parse.add_argument('--dataset', type=str, choices=['PU-GAN', 'Sketchfab', 'PU1K', 'PUNet_50k', 'PUNet_10k'])
args = parse.parse_args()

file_list = 'data/evaluate/%s/evaluatelist.txt' % args.dataset
gt_poisson_dir = 'data/evaluate/%s/gt_poisson' % args.dataset
mesh_dir = 'data/evaluate/%s/normalized_meshes' % args.dataset

metrics = ['cd', 'hd', 'p2f']

for m in metrics:
    if m == 'p2f':
        evaluate(m, file_list, args.outputdir, mesh_dir)
    else:
        evaluate(m, file_list, args.outputdir, gt_poisson_dir)
