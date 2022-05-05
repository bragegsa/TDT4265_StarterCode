import torch.nn as nn
import torch
import math
import torch.nn.functional as F

# Useful links:
# https://blog.krybot.com/a?ID=01000-2f67224c-e217-47e7-ba20-490fbe4a65d9
# https://pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html

def calculate_focal_loss(loss, labels, alpha, gamma=2):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_priors): the loss for each example.
        labels (N, num_priors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """

    pk = F.softmax(loss, dim=1)
    one_hot_encoded = F.one_hot(labels, num_classes=loss.shape[1]).transpose(1,2)
    alpha = torch.tensor(alpha).reshape((1, 9, 1)).to(pk.device)

    # FL = -ak * (1-pk)^y * y * log(pk)
    # focal = -alpha * torch.pow(1.0-pk, gamma) * gamma * torch.log(pk)
    focal = -alpha * torch.pow(1.0-pk, gamma) * torch.log(pk)
    loss_encoded = one_hot_encoded * focal
    focal_loss = loss_encoded.sum(dim=1).mean()

    return focal_loss


class FocalLoss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """
    def __init__(self, anchors, alpha):
        super().__init__()
        self.scale_xy = 1.0/anchors.scale_xy
        self.scale_wh = 1.0/anchors.scale_wh

        self.alpha = alpha
        # print("self.alpha:", self.alpha)

        self.sl1_loss = nn.SmoothL1Loss(reduction='none')
        self.anchors = nn.Parameter(anchors(order="xywh").transpose(0, 1).unsqueeze(dim = 0),
            requires_grad=False)


    def _loc_vec(self, loc):
        """
            Generate Location Vectors
        """
        gxy = self.scale_xy*(loc[:, :2, :] - self.anchors[:, :2, :])/self.anchors[:, 2:, ]
        gwh = self.scale_wh*(loc[:, 2:, :]/self.anchors[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()
    
    def forward(self,
            bbox_delta: torch.FloatTensor, confs: torch.FloatTensor,
            gt_bbox: torch.FloatTensor, gt_labels: torch.LongTensor):
        """
        NA is the number of anchor boxes (by default this is 8732)
            bbox_delta: [batch_size, 4, num_anchors]
            confs: [batch_size, num_classes, num_anchors]
            gt_bbox: [batch_size, num_anchors, 4]
            gt_label = [batch_size, num_anchors]
        """
        gt_bbox = gt_bbox.transpose(1, 2).contiguous() # reshape to [batch_size, 4, num_anchors]

        # Claculating focal loss:
        focal_loss = calculate_focal_loss(confs, gt_labels, self.alpha) 

        pos_mask = (gt_labels > 0).unsqueeze(1).repeat(1, 4, 1)
        bbox_delta = bbox_delta[pos_mask]
        gt_locations = self._loc_vec(gt_bbox)
        gt_locations = gt_locations[pos_mask]
        regression_loss = F.smooth_l1_loss(bbox_delta, gt_locations, reduction="sum")
        num_pos = gt_locations.shape[0]/4
        total_loss = regression_loss/num_pos + focal_loss
        classification_loss = focal_loss
        to_log = dict(
            regression_loss=regression_loss/num_pos,
            classification_loss=focal_loss,
            total_loss=total_loss
        )
        return total_loss, to_log
