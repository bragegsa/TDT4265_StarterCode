from .resnetFPN_2_config import (
    train,
    optimizer,
    schedulers,
    # loss_objective,
    model,
    backbone,
    data_train,
    data_val,
    train_cpu_transform,
    val_cpu_transform,
    gpu_transform,
    label_map,
    anchors
)
from tops.config import LazyCall as L
from ssd.modeling import FocalLoss

# loss_objective = L(FocalLoss)(anchors="${anchors}", alpha=[0.01,*[1 for i in range(model.num_classes-1)]])
loss_objective = L(FocalLoss)(anchors="${anchors}", alpha=[10,*[1000 for i in range(model.num_classes-1)]])