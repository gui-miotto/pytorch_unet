import torch

from deep_segm.unet.contraction import Contraction


class ContractingPath(torch.nn.Module):

    def __init__(self, n_contractions: int, in_channel_size: int):
        super(ContractingPath, self).__init__()

        self.n_contractions = n_contractions
        self.in_channel_size = in_channel_size

        # Layers
        contractions = list()
        for path_index in range(self.n_contractions):
            contract = Contraction(
                in_channels=2 ** (path_index + 5) if path_index > 0 else 1,
                in_channel_size=in_channel_size,
            )
            contractions.append(contract)
            in_channel_size = contract.out_channel_size

        self.out_channel_size = contractions[-1].out_channel_size
        self.contractions = torch.nn.ModuleList(contractions)

    @property
    def in_channels(self):
        return self.contractions[0].in_channels

    @property
    def out_channels(self):
        return self.contractions[-1].out_channels

    def forward(self, x):
        ys = list()
        for contraction in self.contractions:
            x, y = contraction(x)
            # y is the tensor that is going to be send to a expanding module via the skip
            # connections. It is already cropped to the right size.
            ys.append(y)
        # Reverse the ys list since the last output will the first to be consumed by
        # the expanding path.
        ys = ys[::-1]
        return x, ys
