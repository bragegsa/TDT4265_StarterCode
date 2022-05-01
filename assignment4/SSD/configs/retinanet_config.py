from ssd.modeling import AnchorBoxes, SSD300
from ssd.modeling.backbones import Resnet50WithFPN
from tops.config import LazyCall as L
from ssd.modeling.backbones import BasicModel
# The line belows inherits the configuration set for the tdt4265 dataset
from .tdt4265 import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    # model,
    # backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map
)

import torchvision.models as models

# The config below is copied from the ssd300.py model trained on images of size 300*300.
# The images in the tdt4265 dataset are of size 128 * 1024, so resizing to 300*300 is probably a bad idea
# Change the imshape to (128, 1024) and experiment with better prior boxes
train.imshape = (128, 1024)

# backbone = L(BasicModel)(
backbone = L(Resnet50WithFPN)(
    output_channels = [256, 512, 1024, 2048, 64, 64],
    # output_channels = [256,256,256,2048,64,64],
    image_channels = "${train.image_channels}",
    output_feature_sizes = "${anchors.feature_sizes}"
)

model = L(SSD300)(
    feature_extractor="${backbone}",
    anchors="${anchors}",
    loss_objective="${loss_objective}",
    num_classes=8 + 1  # Add 1 for background
)  

anchors = L(AnchorBoxes)(
    feature_sizes= [[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides= [[4, 4], [8, 8], [16, 16], [32, 32], [64, 64], [128, 128]],
    min_sizes= [[16, 16], [32, 32], [48, 48], [64, 64], [86, 86], [128, 128], [128, 400]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    # aspect ratio is defined per feature map (first index is largest feature map (38x38))
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios= [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)