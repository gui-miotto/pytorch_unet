import torch

from deep_segm.unet.expansion import Expansion


class ExpandingPath(torch.nn.Module):

    def __init__(self, n_expansions: int, in_channel_size: int):
        super(ExpandingPath, self).__init__()

        # The number of expansions has to be the same as the number of contractions
        # in the contracting path, otherwise the Tensor shapes won't match.
        self.n_expansions = n_expansions
        self.in_channel_size = in_channel_size

        # Layers
        expansions = list()
        for path_index in range(self.n_expansions):
            exp = Expansion(
                in_channels=2 ** (self.n_expansions + 6 - path_index),
                in_channel_size=in_channel_size,
                )
            expansions.append(exp)
            in_channel_size = exp.out_channel_size

        self.out_channel_size = expansions[-1].out_channel_size
        self.expansions = torch.nn.ModuleList(expansions)

    @property
    def in_channels(self):
        return self.expansions[0].in_channels

    @property
    def out_channels(self):
        return self.expansions[-1].out_channels

    def forward(self, x, from_contractions):
        for y, exp in zip(from_contractions, self.expansions):
            x = exp(x, y)
        return x
