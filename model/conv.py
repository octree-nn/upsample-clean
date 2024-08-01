import torch
from typing import List
from ocnn.nn import OctreeConv, OctreeDeconv, OctreeGroupNorm
from ocnn.octree import Octree


class OctreeConvGnRelu(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, group: int, 
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.stride = stride
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    if self.stride == 2:
      depth = depth - 1
    out = self.gn(out, octree, depth)
    out = self.relu(out)
    return out
  

class OctreeDeconvGnRelu(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, group: int,
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.stride = stride
    self.deconv = OctreeDeconv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.deconv(data, octree, depth)
    if self.stride == 2:
      depth = depth + 1
    out = self.gn(out, octree, depth)
    out = self.relu(out)
    return out


class Conv1x1(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, use_bias: bool = False):
    super().__init__()
    self.linear = torch.nn.Linear(in_channels, out_channels, use_bias)

  def forward(self, data: torch.Tensor):
    r''''''

    return self.linear(data)


class Conv1x1GnRelu(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, group: int, nempty: bool = False):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)
    self.relu = torch.nn.ReLU(inplace=True)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data)
    out = self.gn(out, octree, depth)
    out = self.relu(out)
    return out

class Conv1x1Gn(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, group: int, nempty: bool = False):
    super().__init__()
    self.conv = Conv1x1(in_channels, out_channels, use_bias=False)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data)
    out = self.gn(out, octree, depth)
    return out
  
class OctreeConvGn(torch.nn.Module):
  def __init__(self, in_channels: int, out_channels: int, group: int, 
               kernel_size: List[int] = [3], stride: int = 1,
               nempty: bool = False):
    super().__init__()
    self.conv = OctreeConv(
        in_channels, out_channels, kernel_size, stride, nempty)
    self.gn = OctreeGroupNorm(out_channels, group=group, nempty=nempty)

  def forward(self, data: torch.Tensor, octree: Octree, depth: int):
    r''''''

    out = self.conv(data, octree, depth)
    out = self.gn(out, octree, depth)
    return out
  