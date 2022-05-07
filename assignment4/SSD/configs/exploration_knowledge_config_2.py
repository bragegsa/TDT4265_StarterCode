from .retinanet_init_weights_config import (
    train,
    optimizer,
    schedulers,
    loss_objective,
    model,
    backbone,
    # data_train,
    # data_val,
    # train_cpu_transform,
    # val_cpu_transform,
    gpu_transform,
    label_map,
)
from tops.config import LazyCall as L
from ssd.modeling import AnchorBoxes
from .tdt4265_augmented_2_config import train_cpu_transform, val_cpu_transform, data_train, data_val

anchors = L(AnchorBoxes)(
    # Parameters:
    #     feature_sizes:  Dimensions of the anchor box placement grids
    #     aspect_ratios:  Defines two boxes per element in the list, corresponding to maps in feature_maps
    #                     For example, if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    #     strides:        The number of pixels between each spatial position in the feature map
    #     min_sizes:      The smallest sizes for the anchor box
    #
    # [vertical, horizontal], counting from bottom up and left to right

    feature_sizes = [[32, 256],     [16, 128],  [8, 64],    [4, 32],    [2, 16],    [1, 8]],
    aspect_ratios = [[2],           [2],        [2],        [2],        [2],        [2]],
    strides =       [[4, 4],        [8, 8],     [16, 16],   [32, 32],   [64, 64],   [128, 128]],
    # These min_sizes are pretty decent
    min_sizes =     [[8, 8],        [16, 16],   [32, 32],   [48, 48],   [64, 64],   [128, 128],     [128, 1024]],
    image_shape = "${train.imshape}",
    scale_center_variance = 0.1,
    scale_size_variance = 0.2
)
