from .retinanet_init_weights_config import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    # anchors
)
from tops.config import LazyCall as L
from ssd.modeling import AnchorBoxes

anchors = L(AnchorBoxes)(
    feature_sizes=[[32, 256], [16, 128], [8, 64], [4, 32], [2, 16], [1, 8]],
    # Strides is the number of pixels (in image space) between each spatial position in the feature map
    strides=[[4, 3], [8, 5], [16, 11], [32, 21], [64, 43], [128, 64]],
    min_sizes=[[4, 4], [16, 16], [48, 48], [64, 64], [86, ], [85, 85], [128, 128]],
    # aspect ratio is defined per feature map (first index is largest feature map)
    # aspect ratio is used to define two boxes per element in the list.
    # if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    # Number of boxes per location is in total 2 + 2 per aspect ratio
    aspect_ratios=[[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],
    image_shape="${train.imshape}",
    scale_center_variance=0.1,
    scale_size_variance=0.2
)
