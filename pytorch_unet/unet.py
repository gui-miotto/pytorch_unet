import torch

from deep_segm.unet.conv_train import ConvTrain
from deep_segm.unet.contracting_path import ContractingPath
from deep_segm.unet.expanding_path import ExpandingPath


class UNet(torch.nn.Module):

    @classmethod
    def load(cls, model_path: str):
        model = torch.load(model_path, map_location=torch.device("cpu"))
        model = model.module if isinstance(model, torch.nn.DataParallel) else model
        return model

    def __init__(self, n_classes:int, image_size: int):
        super(UNet, self).__init__()

        if not isinstance(image_size, int) or image_size % 2 != 0 or image_size < 32:
            raise Exception("image_size should be an even integer greater or equal 32.")

        self.n_transforms = 4
        self.n_classes = n_classes
        self.image_size = image_size
        self.in_channel_size = image_size

        # Layers
        self.contract = ContractingPath(
            n_contractions=self.n_transforms,
            in_channel_size=image_size,
        )
        self.valley_conv = ConvTrain(
            in_channels=self.contract.out_channels,
            out_channels=self.contract.out_channels * 2,
            in_channel_size=self.contract.out_channel_size
        )
        self.expand = ExpandingPath(
            n_expansions=self.n_transforms,
            in_channel_size=self.valley_conv.out_channel_size,
        )
        self.final_conv = torch.nn.Conv2d(
            in_channels=self.expand.out_channels,
            out_channels=self.n_classes,
            kernel_size=1,
        )

        # The last convolution uses a 1x1 kernel, therefore it does not change the channel size.
        self.out_channel_size = self.expand.out_channel_size
        self.in_to_out_crop = self._setup_crop_slices()

    def _setup_crop_slices(self):
        """ Calculate the cropping slices that have to used in the contracting path such that
        its outputs are 'concatanable' with the inputs of the expanding path.

        It also returns the cropping slice that when applied to the input to make it matches
        the size of the network's out. This is going to be used to crop the label segmentation
        map (ground truth) such that it is possible to compare it with the network's prediction.
        """
        # Here I'm zipping the contraction/expansion pairs that communicate with each other,
        # therefore one of the paths has to be reversed. Let's reverse the expanding one.
        for cntr, exp in zip(self.contract.contractions, self.expand.expansions[::-1]):
            slack = (cntr.tap_channel_size - exp.tap_channel_size) // 2
            cntr.crop_slice = slice(slack, slack + exp.tap_channel_size)  # Don't use -slack as end

        in_out_slack = (self.in_channel_size - self.out_channel_size) // 2
        in_to_out_crop = slice(in_out_slack, in_out_slack + self.out_channel_size)
        return in_to_out_crop

    def forward(self, x):
        x, ys = self.contract(x)
        x = self.valley_conv(x)
        x = self.expand(x, ys)
        x = self.final_conv(x)
        return x
