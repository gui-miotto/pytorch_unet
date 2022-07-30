import torch

from deep_segm.unet.conv_train import ConvTrain


class Contraction(torch.nn.Module):

    def __init__(self,  in_channels: int, in_channel_size: int):
        super(Contraction, self).__init__()

        # If in_channels == 1, it is the input layer that will receive the image pixel values
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if in_channels > 1 else 64
        self.crop_slice = None  # will be set externally

        # Layers
        self.conv = ConvTrain(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            in_channel_size=in_channel_size,
        )
        self.pool = torch.nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.in_channel_size = in_channel_size
        self.tap_channel_size = self.conv.out_channel_size
        self.out_channel_size = self.conv.out_channel_size // 2  # As the result of pooling

    def forward(self, x):
        # The result of this convolution is what I'm calling tap
        x = self.conv(x)
        # The tap has to be cropped to be used by the expanding path
        y = x[:, :, self.crop_slice, self.crop_slice].clone()
        # The result of the pooling will be used by the next contraction in the contracting path
        x = self.pool(x)
        return x, y
