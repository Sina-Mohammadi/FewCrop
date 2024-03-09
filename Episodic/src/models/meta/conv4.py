import torch.nn as nn
from .metamodules import (MetaModule, MetaSequential, MetaConv1d,
                          MetaBatchNorm1d, MetaLinear)



from torch import nn



import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PyramidPooling(nn.Module):
    def __init__(self, levels, mode="avg"):
        """
        General Pyramid Pooling class which uses Spatial Pyramid Pooling by default and holds the static methods for both spatial and temporal pooling.
        :param levels defines the different divisions to be made in the width and (spatial) height dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n], where  n: sum(filter_amount*level*level) for each level in levels (spatial) or
                                                                    n: sum(filter_amount*level) for each level in levels (temporal)
                                            which is the concentration of multi-level pooling
        """
        super(PyramidPooling, self).__init__()
        self.levels = levels
        self.mode = mode

    def forward(self, x):
        return self.spatial_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        out = 0
        for level in self.levels:
            out += filters * level * level
        return out


    @staticmethod
    def temporal_pyramid_pool(previous_conv, out_pool_size, mode):
        """
        Static Temporal Pyramid Pooling method, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param previous_conv input tensor of the previous convolutional layer
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        num_sample = previous_conv.size(0)
        previous_conv_size = [int(previous_conv.size(2)), int(previous_conv.size(3))]
        for i in range(len(out_pool_size)):
            # print(previous_conv_size)
            #
            h_kernel = previous_conv_size[0]
            w_kernel = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
            w_pad1 = int(math.floor((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            w_pad2 = int(math.ceil((w_kernel * out_pool_size[i] - previous_conv_size[1]) / 2))
            assert w_pad1 + w_pad2 == (w_kernel * out_pool_size[i] - previous_conv_size[1])

            padded_input = F.pad(input=previous_conv, pad=[w_pad1, w_pad2],
                                 mode='constant', value=0)
            if mode == "max":
                pool = nn.MaxPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            elif mode == "avg":
                pool = nn.AvgPool2d((h_kernel, w_kernel), stride=(h_kernel, w_kernel), padding=(0, 0))
            else:
                raise RuntimeError("Unknown pooling type: %s, please use \"max\" or \"avg\".")
            x = pool(padded_input)
            if i == 0:
                tpp = x.view(num_sample, -1)
            else:
                tpp = torch.cat((tpp, x.view(num_sample, -1)), 1)

        return tpp

class TemporalPyramidPooling(PyramidPooling):
    def __init__(self, levels, mode="avg"):
        """
        Temporal Pyramid Pooling Module, which divides the input Tensor horizontally (last dimensions)
        according to each level in the given levels and pools its value according to the given mode.
        Can be used as every other pytorch Module and has no learnable parameters since it's a static pooling.
        In other words: It divides the Input Tensor in "level" horizontal stripes with width of roughly (previous_conv.size(3) / level)
        and the original height and pools the values inside this stripe
        :param levels defines the different divisions to be made in the width dimension
        :param mode defines the underlying pooling mode to be used, can either be "max" or "avg"
        :returns (forward) a tensor vector with shape [batch x 1 x n],
                                            where n: sum(filter_amount*level) for each level in levels
                                            which is the concentration of multi-level pooling
        """
        super(TemporalPyramidPooling, self).__init__(levels, mode=mode)

    def forward(self, x):
        return self.temporal_pyramid_pool(x, self.levels, self.mode)

    def get_output_size(self, filters):
        """
        Calculates the output shape given a filter_amount: sum(filter_amount*level) for each level in levels
        Can be used to x.view(-1, tpp.get_output_size(filter_amount)) for the fully-connected layers
        :param filters: the amount of filter of output fed into the temporal pyramid pooling
        :return: sum(filter_amount*level)
        """
        out = 0
        for level in self.levels:
            out += filters * level
        return out








def conv(in_channels, out_channels, **kwargs):
    return MetaSequential(
        MetaConv1d(in_channels, out_channels, kernel_size=7, **kwargs),
        MetaBatchNorm1d(out_channels, momentum=1., track_running_stats=False),
        nn.ReLU(),
    )


class conv4(MetaModule):
    def __init__(self, num_classes, in_channels=9, hidden_size=None, **kwargs):
        super(conv4, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.features = MetaSequential(
            conv(9, 128),
            conv(128, 128),
            conv(128, 128),
            conv(128, 128)
        )
        self.poool= PyramidPooling([1,4,16],mode='avg')

        self.classifier = MetaLinear(2688, num_classes)#num_classes

    def forward(self, inputs, params=None, features=False):
        x = self.features(inputs, params=self.get_subdict(params, 'features'))
        #x = x.view((x.size(0), -1))
        x = x.view(x.size(0), x.size(1), 1 , x.size(2))
        x=self.poool.temporal_pyramid_pool(x,out_pool_size=[1,4,16],mode='avg')
        if features:
            return x
        logits = self.classifier(x, params=self.get_subdict(params, 'classifier'))
        return logits