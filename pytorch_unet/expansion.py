import torch

from deep_segm.unet.conv_train import ConvTrain

class Expansion(torch.nn.Module):
    def __init__(self, in_channels: int, in_channel_size: int):
        super(Expansion, self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels // 2
        self.in_channel_size = in_channel_size

        # Layers
        self.up_conv = torch.nn.ConvTranspose2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=2,
            stride=2,
        )
        self.tap_channel_size = in_channel_size * 2  # As a result of the up_conv
        self.conv_train = ConvTrain(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            in_channel_size=self.tap_channel_size,  # As a result of the up_conv
        )
        self.out_channel_size = self.conv_train.out_channel_size

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """forward

        Parameters
        ----------
        x : torch.Tensor
            Output of the previous module of the FCN (fully convolutional network) path.
        y : torch.Tensor
            Output of the corresponding contraction module in the contracting path.
        """
        x = self.up_conv(x)
        x = torch.cat((x, y), dim=1)
        x = self.conv_train(x)
        return x
