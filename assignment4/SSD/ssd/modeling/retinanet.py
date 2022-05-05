import torch
import torch.nn as nn
from .anchor_encoder import AnchorEncoder
from torchvision.ops import batched_nms


class RetinaNet(nn.Module):
    def __init__(self, 
            feature_extractor: nn.Module,
            anchors,
            loss_objective,
            num_classes: int,
            anchor_prob_initialization: bool):
        super().__init__()
        """
            Implements the SSD network.
            Backbone outputs a list of features, which are gressed to SSD output with regression/classification heads.
        """

        self.feature_extractor = feature_extractor
        self.loss_func = loss_objective
        self.num_classes = num_classes
        self.regression_heads = []
        self.classification_heads = []
        self.anchor_prob_initialization = anchor_prob_initialization
        self.num_anchors_last = anchors.num_boxes_per_fmap[-1]
        print("self.num_anchors_last:", self.num_anchors_last)

        # Initialize output heads that are applied to each feature map from the backbone.
        for n_boxes, out_ch in zip(anchors.num_boxes_per_fmap, self.feature_extractor.out_channels):
            
            # print("anchors.num_boxes_per_fmap:", anchors.num_boxes_per_fmap)
            # print("n_boxes:", n_boxes)
            map_1 = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, n_boxes * 4, kernel_size=3, padding=1)
                )

            map2 = nn.Sequential(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(out_ch, n_boxes * self.num_classes, kernel_size=3, padding=1)
                )

            # self.regression_heads.append(nn.Conv2d(out_ch, n_boxes * 4, kernel_size=3, padding=1))
            self.regression_heads.append(map_1)
            # self.classification_heads.append(nn.Conv2d(out_ch, n_boxes * self.num_classes, kernel_size=3, padding=1))
            self.classification_heads.append(map2)

        self.regression_heads = nn.ModuleList(self.regression_heads)
        self.classification_heads = nn.ModuleList(self.classification_heads)
        self.anchor_encoder = AnchorEncoder(anchors)
        self._init_weights()

    def _init_weights(self):
        layers = [*self.regression_heads, *self.classification_heads]
        if(self.anchor_prob_initialization):
            print(" --- Weights initialized ---")
            p = 0.99 # Used in the tip
            K = self.num_classes
            sigma = 0.01
            b = 0
            pi = 0.01
            b_background = torch.log(torch.tensor(p*(K-1)/(1-p)))
            b_final = -torch.log(torch.tensor((1-pi)/pi))
            
            for layer in layers:
                for param in layer:
                    if hasattr(param, "bias"):
                        if len(param.bias)%9 == 0:
                            print("End layer length:", len(param.bias))
                            nn.init.normal_(param.bias.data[:], b, sigma)
                            nn.init.constant_(param.bias.data[:int(len(param.bias)/9)], b_background)
                        else:
                            nn.init.normal_(param.bias.data[:], b, sigma)
                        # print("param.bias.data[:]", param.bias.data[:])
                        # print("length", len(param.bias))
                        # nn.init.normal_(param.bias.data[:], b, sigma)
                        # nn.init.constant_(param.bias.data[:self.num_anchors_last], b_background) # A bit unsure about this one

            # print("self.classification_heads[-1][-1].bias", self.classification_heads[-1][-1].bias)
            nn.init.constant_(self.classification_heads[-1][-1].bias, b_final)
            # print("self.classification_heads[-1][-1].bias", self.classification_heads[-1][-1].bias)
            nn.init.constant_(self.classification_heads[-1][-1].bias.data[:self.num_anchors_last], b_background)
            # print("self.classification_heads[-1][-1].bias", self.classification_heads[-1][-1].bias)

        else:
            print(" --- Weights not initialized ---")
            for layer in layers:
                print("Layer:", layer)
                for param in layer.parameters():
                    if param.dim() > 1: nn.init.xavier_uniform_(param)

    def regress_boxes(self, features):
        locations = []
        confidences = []
        for idx, x in enumerate(features):
            bbox_delta = self.regression_heads[idx](x).view(x.shape[0], 4, -1)
            bbox_conf = self.classification_heads[idx](x).view(x.shape[0], self.num_classes, -1)
            locations.append(bbox_delta)
            confidences.append(bbox_conf)
        bbox_delta = torch.cat(locations, 2).contiguous()
        confidences = torch.cat(confidences, 2).contiguous()
        return bbox_delta, confidences

    
    def forward(self, img: torch.Tensor, **kwargs):
        """
            img: shape: NCHW
        """
        if not self.training:
            return self.forward_test(img, **kwargs)
        features = self.feature_extractor(img)
        return self.regress_boxes(features)
    
    def forward_test(self,
            img: torch.Tensor,
            imshape=None,
            nms_iou_threshold=0.5, max_output=200, score_threshold=0.05):
        """
            img: shape: NCHW
            nms_iou_threshold, max_output is only used for inference/evaluation, not for training
        """
        features = self.feature_extractor(img)
        bbox_delta, confs = self.regress_boxes(features)
        boxes_ltrb, confs = self.anchor_encoder.decode_output(bbox_delta, confs)
        predictions = []
        for img_idx in range(boxes_ltrb.shape[0]):
            boxes, categories, scores = filter_predictions(
                boxes_ltrb[img_idx], confs[img_idx],
                nms_iou_threshold, max_output, score_threshold)
            if imshape is not None:
                H, W = imshape
                boxes[:, [0, 2]] *= H
                boxes[:, [1, 3]] *= W
            predictions.append((boxes, categories, scores))
        return predictions

 
def filter_predictions(
        boxes_ltrb: torch.Tensor, confs: torch.Tensor,
        nms_iou_threshold: float, max_output: int, score_threshold: float):
        """
            boxes_ltrb: shape [N, 4]
            confs: shape [N, num_classes]
        """
        assert 0 <= nms_iou_threshold <= 1
        assert max_output > 0
        assert 0 <= score_threshold <= 1
        scores, category = confs.max(dim=1)

        # 1. Remove low confidence boxes / background boxes
        mask = (scores > score_threshold).logical_and(category != 0)
        boxes_ltrb = boxes_ltrb[mask]
        scores = scores[mask]
        category = category[mask]

        # 2. Perform non-maximum-suppression
        keep_idx = batched_nms(boxes_ltrb, scores, category, iou_threshold=nms_iou_threshold)

        # 3. Only keep max_output best boxes (NMS returns indices in sorted order, decreasing w.r.t. scores)
        keep_idx = keep_idx[:max_output]
        return boxes_ltrb[keep_idx], category[keep_idx], scores[keep_idx]