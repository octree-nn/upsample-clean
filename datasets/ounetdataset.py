import torch
import numpy as np
import os 

from ocnn.octree import Octree, Points
from ocnn.dataset import CollateBatch
from thsolver import Dataset
from .data_transforms import Compose


class ReadFile:
    def __init__(self, root):
        self.root = root  # 训练集 or 测试集的路径 'xxx/PCN/train' or 'xxx/PCN/test'

    def __call__(self, filename):
        input_path = os.path.join(self.root, 'input', filename)
        if 'cleaning' in filename:
          filename = filename.split('/')
          gt_path = os.path.join(self.root, 'gt', filename[0], filename[-1])
        else:
          gt_path = os.path.join(self.root, 'gt', filename)
        input = np.load(input_path)
        gt = np.load(gt_path)
        return {'input': input, 'gt': gt}

class Transform:

  def __init__(self, flags):
    super().__init__()
    self.noise_level = flags.noise_level
    self.depth = flags.depth
    self.full_depth = flags.full_depth
    self.augment = flags.data_augment
    self.is_train = flags.is_train
    self.random_mirror = self.random_mirror_points_train()

  def points2octree(self, points: Points):
    octree = Octree(self.depth, self.full_depth)
    octree.build_octree(points)
    return octree
  
  def random_mirror_points_train(self):
    return Compose([{
              'callback': 'RandomMirrorPoints',
              'objects': ['input', 'gt']}])

  def __call__(self, point_cloud, idx):
    pcd_gt, pcd_in = point_cloud['gt'], point_cloud['input']

    # process input point cloud
    if self.is_train and (pcd_in.shape[0] == 50000 or pcd_in.shape[0] == 10000):
      radius = np.max(np.linalg.norm(pcd_in, axis=1)) 
      noise_level = np.random.uniform(self.noise_level[0], self.noise_level[1], size=(1, ))
      pcd_in = pcd_in + np.random.normal(0, noise_level * radius, pcd_in.shape) 
  
    if pcd_in.shape[0] == 50000 or pcd_in.shape[0] == 10000:
      feature = np.ones_like(pcd_in[:, :1])
    else:
      feature = np.zeros_like(pcd_in[:, :1])
    feature = torch.from_numpy(feature).float()

    if self.augment:
      data = {'input': pcd_in, 'gt': pcd_gt}
      data = self.random_mirror(data)
      pcd_in, pcd_gt = data['input'], data['gt']
      elastic_scale = np.random.uniform(0.9, 1.1, size=(1, 3))  
      scale = np.random.uniform(0.95, 1.05, size=(1,))
      pcd_in, pcd_gt = pcd_in * elastic_scale * scale, pcd_gt * elastic_scale * scale

    # transform points to octree
    points_in = Points(torch.from_numpy(pcd_in * 1.7).float(), features=feature)
    points_in.clip(min=-1, max=1)
    octree_in = self.points2octree(points_in)

    points_gt = Points(torch.from_numpy(pcd_gt * 1.7).float())
    points_gt.clip(min=-1, max=1)
    octree_gt = self.points2octree(points_gt)
    return {'octree': octree_in,  'points': points_in,
             'octree_gt': octree_gt, 'points_gt': points_gt}

def get_upsample_clean_dataset(flags):
  transform = Transform(flags)
  read_file = ReadFile(flags.location)  
  collate_batch = CollateBatch(merge_points=False)
  dataset = Dataset(flags.location, flags.filelist, transform, read_file)
  return dataset, collate_batch
