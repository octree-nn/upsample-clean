import os
import torch
import ocnn
import numpy as np
import torch.nn.functional as F

from datasets import get_upsample_clean_dataset
from thsolver import Solver, get_config
from model import OUNet


class CompletionSolver(Solver):

  def get_model(self, flags):
    return OUNet(flags)

  def get_dataset(self, flags):
    return get_upsample_clean_dataset(flags)

  def model_forward(self, batch):
    octree_in = batch['octree'].cuda(non_blocking=True)
    octree_gt = batch['octree_gt'].cuda(non_blocking=True)
    model_out = self.model(octree_in, octree_gt, update_octree=False)
    output = self.compute_loss(octree_gt, model_out)
    return output

  def train_step(self, batch):
    output = self.model_forward(batch)
    output = {'train/' + key: val for key, val in output.items()}
    return output

  def test_step(self, batch):
    output = self.model_forward(batch)
    output = {'test/' + key: val for key, val in output.items()}
    return output

  def eval_step(self, batch):
    # forward the model
    octree_in = batch['octree'].cuda(non_blocking=True)
    output = self.model(octree_in, update_octree=True)
    points_out = self.octree2pts(output['octree_out'])

    # save the output point clouds
    filenames = batch['filename']

    for i, filename in enumerate(filenames):
      pos = filename.rfind('.')
      if pos != -1: filename = filename[:pos]  # remove the suffix
      filename_out = os.path.join(self.logdir, 'model_outputs', filename + '.out.xyz')
      os.makedirs(os.path.dirname(filename_out), exist_ok=True)

      # NOTE: it consumes much time to save point clouds to hard disks
      np.savetxt(filename_out, points_out[i].cpu().numpy(), fmt='%.6f')

  def get_ground_truth_signal(self, octree):
    octree_feature = ocnn.modules.InputFeature('L', nempty=True)  
    data = octree_feature(octree)
    return data

  def compute_loss(self, octree: ocnn.octree.Octree, model_out: dict):
    # octree splitting loss
    output = dict()
    logits = model_out['logits']
    for d in logits.keys():
      logitd = logits[d]
      label_gt = octree.nempty_mask(d).long()
      output['loss_%d' % d] = F.cross_entropy(logitd, label_gt)
      output['accu_%d' % d] = logitd.argmax(1).eq(label_gt).float().mean()

    # octree regression loss
    signal = model_out['signal']
    signal_gt = self.get_ground_truth_signal(octree)
    output['loss_reg'] = torch.mean(torch.sum((signal_gt - signal)**2, dim=1))

    # total loss
    losses = [val for key, val in output.items() if 'loss' in key]
    output['loss'] = torch.sum(torch.stack(losses))
    return output

  def octree2pts(self, octree: ocnn.octree.Octree):  
    depth = octree.depth
    batch_size = octree.batch_size

    signal = octree.features[depth]
    displacement = signal

    x, y, z, _ = octree.xyzb(depth, nempty=True)
    xyz = torch.stack([x, y, z], dim=1) + 0.5 + displacement
    xyz = xyz / 2**(depth - 1) - 1.0  # [0, 2^depth] -> [-1, 1]

    point_cloud = xyz
    batch_id = octree.batch_id(depth, nempty=True)
    points_num = [torch.sum(batch_id == i) for i in range(batch_size)]
    points = torch.split(point_cloud, points_num)
    return points

  def load_checkpoint(self):
    ckpt = self.FLAGS.SOLVER.ckpt
    if not ckpt:
      # If ckpt is empty, then get the latest checkpoint from ckpt_dir
      if not os.path.exists(self.ckpt_dir):
        return
      ckpts = sorted(os.listdir(self.ckpt_dir))
      if len(ckpts) > 0:
        ckpt = os.path.join(self.ckpt_dir, ckpts[-1])
    if not ckpt:
      return  # return if ckpt is still empty

    # load trained model
    trained_dict = torch.load(ckpt, map_location='cuda')
    if ckpt.endswith('.solver.tar'):
      model_dict = trained_dict['model_dict']
      self.start_epoch = trained_dict['epoch'] + 1  # !!! add 1
      if self.optimizer:
        self.optimizer.load_state_dict(trained_dict['optimizer_dict'])
      if self.scheduler:
        self.scheduler.load_state_dict(trained_dict['scheduler_dict'])
    else:
      model_dict = trained_dict
    model = self.model.module if self.world_size > 1 else self.model
    model.load_state_dict(model_dict)

  @classmethod
  def update_configs(cls):
    FLAGS = get_config()

    FLAGS.DATA.train.points_scale = 128
    FLAGS.DATA.test = FLAGS.DATA.train.clone()

    FLAGS.MODEL.depth = 6
    FLAGS.MODEL.full_depth = 2

if __name__ == '__main__':
  CompletionSolver.main()
