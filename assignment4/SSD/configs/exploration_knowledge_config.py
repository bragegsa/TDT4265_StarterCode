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
from .tdt4265_augmented_3_config import train_cpu_transform, val_cpu_transform, data_train, data_val

anchors = L(AnchorBoxes)(
    # Parameters:
    #     feature_sizes:  Dimensions of the anchor box placement grids
    #     aspect_ratios:  Defines two boxes per element in the list, corresponding to maps in feature_maps
    #                     For example, if ratio=[2], boxes will be created with ratio 1:2 and 2:1
    #                     Then the number of boxes per location is in total 2 + 2 per aspect ratio
    #     strides:        The number of pixels between each spatial position in the feature map
    #     min_sizes:      (?) Lower bound for anchor box sizes
    #
    # [vertical, horizontal], counting from bottom up and left to right

    feature_sizes = [[32, 256], [16, 128],  [8, 64],    [4, 32],    [2, 16],    [1, 8]],
    aspect_ratios = [[2, 3],    [2],        [2, 3],     [2, 3],     [2, 3],     [2, 3, 4]],
    strides =       [[4, 3],    [8, 5],     [16, 11],   [32, 21],   [64, 43],   [64, 128]],
    min_sizes =     [[50, 50],  [50, 50],   [50, 50],   [50, 50],   [50, 50],   [50, 50],   [50, 50]],
    image_shape = "${train.imshape}",
    scale_center_variance = 0.1,
    scale_size_variance = 0.2
)
