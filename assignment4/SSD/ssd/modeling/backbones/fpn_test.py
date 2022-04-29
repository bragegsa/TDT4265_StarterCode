import torch
import torchvision
from torch import nn
from typing import Tuple, List
import torchvision.models as models
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork

# https://pytorch.org/vision/main/feature_extraction.html?highlight=backbone_utils
# http://pytorch.org/vision/main/generated/torchvision.models.feature_extraction.create_feature_extractor.html

return_nodes = {
    'layer1': 'layer1',
    'layer2': 'layer2',
    'layer3': 'layer3',
    # 'layer4': 'layer4',
}

# Now you can build the feature extractor. This returns a module whose forward
# method returns a dictionary like:
# {
#     'layer1': output of layer 1,
#     'layer2': output of layer 2,
#     'layer3': output of layer 3,
#     'layer4': output of layer 4,
# }

# Let's put all that together to wrap resnet50 with MaskRCNN

# MaskRCNN requires a backbone with an attached FPN
class FPN(torch.nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        # Get a resnet50 backbone
        m = models.resnet34()
        # Extract 4 main layers (note: MaskRCNN needs this particular name
        # mapping for return nodes)
        self.body = create_feature_extractor(
            m, return_nodes={f'layer{k}': str(v)
                             for v, k in enumerate([1, 2, 3, 4])})
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.body(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        self.out_channels = 256
        self.fpn = FeaturePyramidNetwork(
            in_channels_list, out_channels=self.out_channels,
            extra_blocks=LastLevelMaxPool())

    def forward(self, x):
        x = self.body(x)
        x = self.fpn(x)
        return x


# Now we can build our model!
model = MaskRCNN(FPN(), num_classes=91).eval()