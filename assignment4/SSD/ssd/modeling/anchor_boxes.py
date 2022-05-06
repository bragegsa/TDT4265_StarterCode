# Modified from: https://github.com/lufficc/SSD
import torch
from typing import List
from math import sqrt

# Note on center/size variance:
# This is used for endcoding/decoding the regressed coordinates from the SSD bounding box head to actual locations.
# It's a trick to improve gradients from bounding box regression. Take a look at this post about more info:
# https://leimao.github.io/blog/Bounding-Box-Encoding-Decoding/
class AnchorBoxes(object):
    def __init__(self, 
            image_shape: tuple, 
            feature_sizes: List[tuple], 
            min_sizes: List[int],
            strides: List[tuple],
            aspect_ratios: List[int],
            scale_center_variance: float,
            scale_size_variance: float):
        """Generate SSD anchors Boxes.
            It returns the center, height and width of the anchors. The values are relative to the image size
            Args:
                image_shape: tuple of (image height, width)
                feature_sizes: each tuple in the list is the feature shape outputted by the backbone (H, W)
            Returns:
                anchors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        self.scale_center_variance = scale_center_variance
        self.scale_size_variance = scale_size_variance
        self.num_boxes_per_fmap = [2 + 2*len(ratio) for ratio in aspect_ratios]
        # Calculation method slightly different from paper

        anchors = []
        # Iterate through each feature map of size [fH, fW] with index fidx
        for fidx, [fH, fW] in enumerate(feature_sizes):
            bbox_sizes = []
            # Set minimum box heights and widths with min_sizes relative to image
            h_min = min_sizes[fidx][0] / image_shape[0]
            w_min = min_sizes[fidx][1] / image_shape[1]
            # Append the minimum box sizes to bbox_sizes
            bbox_sizes.append((w_min, h_min))
            # Set maximum box heights and widths as:
            # the square root of the product of current min_sizes and min_sizes in the next feature map
            h_max = sqrt(min_sizes[fidx][0]*min_sizes[fidx+1][0]) / image_shape[0]
            w_max = sqrt(min_sizes[fidx][1]*min_sizes[fidx+1][1]) / image_shape[1]
            # Append the maximum box sizes to bbox_sizes
            bbox_sizes.append((w_max, h_max))
            # Iterate through the aspect ratios r for current feature map
            for r in aspect_ratios[fidx]:
                # Scale height and width to get desired aspect ratio r
                h = h_min*sqrt(r)
                w = w_min/sqrt(r)
                # Add scaled boxes to bbox_sizes
                bbox_sizes.append((w_min/sqrt(r), h_min*sqrt(r)))
                bbox_sizes.append((w_min*sqrt(r), h_min/sqrt(r)))
            
            # Set scales for feature points according to image size and desired stride distances
            scale_y = image_shape[0] / strides[fidx][0]
            scale_x = image_shape[1] / strides[fidx][1]
            # Iterate through anchor boxes in bbox_sizes
            for w, h in bbox_sizes:
                # Iterate through all points i an j in height and width of feature map fidx
                for i in range(fH):
                    for j in range(fW):
                        # Scale centers of anchor boxes
                        cx = (j + 0.5)/scale_x
                        cy = (i + 0.5)/scale_y
                        # Add final scaled centers, widths and heights to anchors
                        anchors.append((cx, cy, w, h))

        self.anchors_xywh = torch.tensor(anchors).clamp(min=0, max=1).float()
        self.anchors_ltrb = self.anchors_xywh.clone()
        self.anchors_ltrb[:, 0] = self.anchors_xywh[:, 0] - 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 1] = self.anchors_xywh[:, 1] - 0.5 * self.anchors_xywh[:, 3]
        self.anchors_ltrb[:, 2] = self.anchors_xywh[:, 0] + 0.5 * self.anchors_xywh[:, 2]
        self.anchors_ltrb[:, 3] = self.anchors_xywh[:, 1] + 0.5 * self.anchors_xywh[:, 3]

    def __call__(self, order):
        if order == "ltrb":
            return self.anchors_ltrb
        if order == "xywh":
            return self.anchors_xywh

    @property
    def scale_xy(self):
        return self.scale_center_variance

    @property
    def scale_wh(self):
        return self.scale_size_variance