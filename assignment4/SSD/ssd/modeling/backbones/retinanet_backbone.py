# https://pytorch.org/vision/main/feature_extraction.html?highlight=backbone_utils
# http://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html
import torch
from torch import nn
from typing import Tuple, List
from torchvision.models import resnet50
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork


class Resnet50WithFPN(torch.nn.Module):
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

        self.fpn = FeaturePyramidNetwork(in_channels_list=output_channels, out_channels=256)
        model = resnet50(pretrained=True)
        modules = list(model.children())[:-2]
        backbone = nn.Sequential(*modules)

        self.conv = backbone[0]
        self.bn1 = backbone[1]
        self.relu = backbone[2]
        self.maxpool = backbone[3]
        self.conv1 = backbone[4]
        self.conv2 = backbone[5]
        self.conv3 = backbone[6]
        self.conv4 = backbone[7]
        
        self.map5 = nn.Sequential(

            # Resolution 3x3
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels[-3], out_channels=128, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(output_channels[3], 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=output_channels[-2], kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.map6 = nn.Sequential(

            # Resolution 1x1
            nn.ReLU(),
            nn.Conv2d(in_channels=output_channels[-2], out_channels=256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=output_channels[-1], kernel_size=2, stride=1, padding=0),
            nn.ReLU()
        )

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

        #out_features = nn.ModuleList
        out_features_keys = ["c1","c2","c3","c4","c5","c6"]
        
        #fpn = torchvision.ops.FeaturePyramidNetwork(self.out_channels, 64)
        
        #First layers of ResNet34 
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv1(x)
        out_features.append(x)
        #print("c1: ", c1.shape)
        x = self.conv2(x)
        out_features.append(x)
        #print("c2: ", c2.shape)
        x = self.conv3(x)
        out_features.append(x)
        #print("c3: ", c3.shape)
        x = self.conv4(x)
        out_features.append(x)

        x = self.map5(x)
        out_features.append(x)
        x = self.map6(x)
        out_features.append(x)

        output_dict = dict(zip(out_features_keys, out_features))
        #print(output_dict)
        #output_dict= {out_features_keys[i]: out_features[i] for i in range(len(out_features_keys))}
        output_fpn = self.fpn(output_dict)
        out_features = output_fpn.values()

        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)