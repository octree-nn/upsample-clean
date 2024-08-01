import torch
import ocnn

from ocnn.octree import Octree
from .conv import OctreeConvGnRelu, OctreeDeconvGnRelu, Conv1x1GnRelu
from .resblock import OctreeConvGnRelu, OctreeResBlocks


class OUNet(torch.nn.Module):
    def __init__(self, flags):
        super().__init__()
        self.channel_in = flags.channel_in
        self.channel_out = flags.channel_out
        self.channels = flags.channels
        self.depth = flags.depth
        self.full_depth = flags.full_depth
        self.group = flags.group
        self.feature = flags.feature
        self.resblk_num = flags.resblk_num
        self.bottleneck = flags.bottleneck

        # encoder 
        self.conv1 = OctreeConvGnRelu(self.channel_in, self.channels[self.depth], group=self.group)
        self.encoder_blks = torch.nn.ModuleList([OctreeResBlocks(
            self.channels[d], self.channels[d], resblk_num=self.resblk_num, group=self.group, nempty=False, bottleneck=self.bottleneck)
            for d in range(self.depth, self.full_depth-1, -1)])
        self.downsample = torch.nn.ModuleList([OctreeConvGnRelu(
            self.channels[d], self.channels[d-1], kernel_size=[2], stride=2, group=self.group, 
            nempty=False) for d in range(self.depth, self.full_depth, -1)])

        # decoder 
        self.upsample = torch.nn.ModuleList([OctreeDeconvGnRelu(
            self.channels[d-1], self.channels[d], kernel_size=[2], stride=2, group=self.group,
            nempty=False) for d in range(self.full_depth+1, self.depth+1)])
        self.decoder_blks = torch.nn.ModuleList([OctreeResBlocks(
            self.channels[d], self.channels[d], resblk_num=self.resblk_num, group=self.group, nempty=False, bottleneck=self.bottleneck)
            for d in range(self.full_depth, self.depth+1)])
        
        # header 
        self.predict = torch.nn.ModuleList([self._make_predict_module(
                                            self.channels[d], 2) for d in range(self.full_depth, self.depth + 1)])
        self.header = self._make_predict_module(self.channels[self.depth], self.channel_out)

    def _make_predict_module(self, channel_in, channel_out=2, num_hidden=64):
        return torch.nn.ModuleList([
            Conv1x1GnRelu(channel_in, num_hidden, group=self.group),
            ocnn.modules.Conv1x1(num_hidden, channel_out, use_bias=True)])

    def get_input_feature(self, octree: Octree):
        r''' Get the input feature from the input `octree`.
        '''

        octree_feature = ocnn.modules.InputFeature(self.feature, nempty=False)
        out = octree_feature(octree)
        assert out.size(1) == self.channel_in
        return out

    def encoder(self, octree):
        r''' The encoder network for extracting heirarchy features. 
        '''

        convs = dict()
        depth, full_depth = self.depth, self.full_depth
        data = self.get_input_feature(octree)
        convs[depth] = self.conv1(data, octree, depth)
        for i, d in enumerate(range(depth, full_depth-1, -1)):
            convs[d] = self.encoder_blks[i](convs[d], octree, d)
            if d > full_depth:
                convs[d-1] = self.downsample[i](convs[d], octree, d)
        return convs

    def decoder(self, convs: dict, octree_in: Octree, octree_out: Octree,
              update_octree: bool = False):
        r''' The decoder network for decode the octree.
        '''

        logits = dict()
        deconv = convs[self.full_depth]
        depth, full_depth = self.depth, self.full_depth
        for i, d in enumerate(range(full_depth, depth + 1)):
            if d > full_depth:
                deconv = self.upsample[i-1](deconv, octree_out, d-1)
                skip = ocnn.nn.octree_align(convs[d], octree_in, octree_out, d)
                deconv = deconv + skip  # output-guided skip connections
            deconv = self.decoder_blks[i](deconv, octree_out, d)

            # predict the splitting label
            logit = self.predict[i][0](deconv, octree_out, depth=d)
            logit = self.predict[i][1](logit)
            logits[d] = logit

            # update the octree according to predicted labels
            if update_octree:
                split = logit.argmax(1).int()

                
                octree_out.octree_split(split, d)
                if d < depth:
                    octree_out.octree_grow(d + 1)

            # predict the signal
            if d == depth:
                signal = self.header[0](deconv, octree_out, depth=d)
                signal = self.header[1](signal)
                signal = 0.5 * torch.tanh(signal)
                signal = ocnn.nn.octree_depad(signal, octree_out, depth)
                if update_octree:
                    octree_out.features[depth] = signal

        return {'logits': logits, 'signal': signal, 'octree_out': octree_out}

    def init_octree(self, octree_in: Octree):
        r''' Initialize a full octree for decoding.
        '''

        device = octree_in.device
        batch_size = octree_in.batch_size
        octree = Octree(self.depth, self.full_depth, batch_size, device)
        for d in range(self.full_depth+1):
            octree.octree_grow_full(depth=d)
        return octree

    def forward(self, octree_in, octree_out=None, update_octree: bool = False):
        r''''''

        if octree_out is None:
            update_octree = True
            octree_out = self.init_octree(octree_in)
        convs = self.encoder(octree_in)
        out = self.decoder(convs, octree_in, octree_out, update_octree)
        return out

