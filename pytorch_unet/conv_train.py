import torch

class ConvTrain(torch.nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        in_channel_size: int,
        dropout_p: float = 0.,
        ):
        super(ConvTrain, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_channel_size = in_channel_size
        self.out_channel_size = in_channel_size - 4  # As the result of two 3x3 valid-padded convol.
        self.dropout_p = dropout_p

        # Layers
        self.conv1 = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="valid",
        )
        self.drop1 = torch.nn.Dropout2d(
            p=dropout_p,
            )
        self.act1 = torch.nn.ReLU()
        # 2nd convolution bundle
        self.conv2 = torch.nn.Conv2d(
            in_channels=self.out_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            padding="valid",
        )
        self.drop2 = torch.nn.Dropout2d(p=dropout_p)
        self.act2 = torch.nn.ReLU()

    def forward(self, x):
        # 1st convolution bundle
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.act1(x)
        # 2nd convolution bundle
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.act2(x)
        return x
