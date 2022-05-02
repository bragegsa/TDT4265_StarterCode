from queue import PriorityQueue
import torch
from torch import nn
from typing import Tuple, List
from torchvision.models import resnet101
from torchvision.models.resnet import BasicBlock
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork
from torchvision.models.feature_extraction import create_feature_extractor

# Useful documentation:
# https://pytorch.org/vision/main/feature_extraction.html?highlight=backbone_utils
# http://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html 

class resnetFPN(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        model = resnet101(pretrained=True)

        self.map1 = nn.Sequential(
            BasicBlock(inplanes = self.out_channels[3], planes = self.out_channels[4], stride = 2,
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.out_channels[3], out_channels=self.out_channels[4], kernel_size=1, stride=2),
                nn.ReLU())),
        )
        
        self.map2 = nn.Sequential(
            BasicBlock(inplanes = self.out_channels[4], planes = self.out_channels[5], stride = 2,
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=self.out_channels[4], out_channels=self.out_channels[5], kernel_size=1, stride=2),
                nn.ReLU())),
        )

        # Modified documentation-code from the first link in the top of the document: ----------

        self.body = create_feature_extractor(
            model, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        inp=torch.randn(1, 3, 128, 1024)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]

        # ----------

        with torch.no_grad():
            x = list(out.values())[-1]
            x = self.map1(x)
            x = self.map2(x)

            in_channels_list.append(x.shape[1])

        self.out_channels = [256, 256, 256, 256, 256, 256]
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels=256)

    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """

        out_features = []
        x = self.body(x)

        x["4"] = self.map1(x["3"])
        x["5"] = self.map2(x["4"])
        
        x = self.fpn(x)

        out_features.extend(x.values())

        # for idx, feature in enumerate(out_features):
        #     out_channel = self.out_channels[idx]
        #     h, w = self.output_feature_shape[idx]
        #     expected_shape = (out_channel, h, w)
        #     assert feature.shape[1:] == expected_shape, \
        #         f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        # assert len(out_features) == len(self.output_feature_shape),\
        #     f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"

        return tuple(out_features)










