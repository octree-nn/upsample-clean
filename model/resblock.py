import torch
import torch.utils.checkpoint

from ocnn.octree import Octree
from ocnn.nn import OctreeMaxPool
from .conv import Conv1x1GnRelu, OctreeConvGnRelu, Conv1x1Gn


class OctreeResBlock(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, group: int, stride: int = 1, 
               bottleneck: int = 4, nempty: bool = False):
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.bottleneck = bottleneck
    self.stride = stride
    channelb = int(out_channels / bottleneck)

    if self.stride == 2:
      self.max_pool = OctreeMaxPool(nempty)
    self.conv1x1a = Conv1x1GnRelu(in_channels, channelb, group)
    self.conv3x3 = OctreeConvGnRelu(channelb, channelb, group, nempty=nempty)
    self.conv1x1b = Conv1x1Gn(channelb, out_channels, group)
    if self.in_channels != self.out_channels:
      self.conv1x1c = Conv1x1Gn(in_channels, out_channels, group)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    if self.stride == 2:
      data = self.max_pool(data, octree, depth)
      depth = depth - 1
    conv1 = self.conv1x1a(data, octree, depth)
    conv2 = self.conv3x3(conv1, octree, depth)
    conv3 = self.conv1x1b(conv2, octree, depth)
    if self.in_channels != self.out_channels:
      data = self.conv1x1c(data, octree, depth)
    out = self.relu(conv3 + data)
    return out


class OctreeResBlocks(torch.nn.Module):
  def __init__(self, in_channels, out_channels, resblk_num, group, bottleneck=4,
               nempty=False, resblk=OctreeResBlock, use_checkpoint=False):
    super().__init__()
    self.resblk_num = resblk_num
    self.use_checkpoint = use_checkpoint
    channels = [in_channels] + [out_channels] * resblk_num

    self.resblks = torch.nn.ModuleList(
        [resblk(channels[i], channels[i+1], group, 1, bottleneck, nempty)
         for i in range(self.resblk_num)])

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    for i in range(self.resblk_num):
      if self.use_checkpoint:
        data = torch.utils.checkpoint.checkpoint(
            self.resblks[i], data, octree, depth)
      else:
        data = self.resblks[i](data, octree, depth)
    return data
