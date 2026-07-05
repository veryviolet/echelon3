import copy
from enum import Enum, unique
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch

from echelon3.utils.bboxes import BBoxes, BBoxType, BBoxesClass, YoloBBoxes

def gather_feature(fmap, index, mask=None, use_transform=False):
    if use_transform:
        # change a (N, C, H, W) tenor to (N, HxW, C) shape
        batch, channel = fmap.shape[:2]
        fmap = fmap.view(batch, channel, -1).permute((0, 2, 1)).contiguous()

    dim = fmap.size(-1)
    index = index.unsqueeze(len(index.shape)).expand(*index.shape, dim)
    fmap = fmap.gather(dim=1, index=index)
    if mask is not None:
        # this part is not called in Res18 dcn COCO
        mask = mask.unsqueeze(2).expand_as(fmap)
        fmap = fmap[mask]
        fmap = fmap.reshape(-1, dim)
    return fmap


class DecodeHeatmaps(torch.nn.Module):

    def __init__(self,
                 bbox_type=BBoxType.PASCAL_VOC,
                 num_classes=2,
                 output_size=(224, 224),
                 **kwargs):
        super(DecodeHeatmaps, self).__init__()
        self.bbox_type = BBoxType(bbox_type)
        self.bbox_class = BBoxesClass[self.bbox_type]
        self.num_classes = num_classes
        self.output_size = output_size

    def forward(self, x):
        return x

    def decode(self, fmap, wh, K=100):
        r"""
        decode output feature map to detection results

        Args:
            fmap(Tensor): output feature map
            wh(Tensor): tensor that represents predicted width-height
            reg(Tensor): tensor that represens regression of center points
            cat_spec_wh(bool): whether apply gather on tensor `wh` or not
            K(int): topk value
        """
        device = fmap.device
        batch, channel, height, width = fmap.shape

        fmap = DecodeHeatmaps.pseudo_nms(fmap)

        scores, index, clses, ys, xs = DecodeHeatmaps.topk_score(fmap, K=K)
        xs_abs = xs.view(batch, K, 1)
        ys_abs = ys.view(batch, K, 1)
        wh_rel = gather_feature(wh, index, use_transform=True)
        wh_rel = wh_rel.reshape(batch, K, 2)
        w_rel = wh_rel[..., 0:1]
        h_rel = wh_rel[..., 1:2]
        w_abs = (w_rel * self.output_size[1]).type(torch.LongTensor).to(fmap.device)
        h_abs = (h_rel * self.output_size[1]).type(torch.LongTensor).to(fmap.device)
        xs_rel = (xs_abs.type(torch.FloatTensor) / self.output_size[1]).to(fmap.device)
        ys_rel = (ys_abs.type(torch.FloatTensor) / self.output_size[0]).to(fmap.device)

        clses = clses.reshape(batch, K, 1).float()
        scores = scores.reshape(batch, K, 1)

        if self.bbox_type == BBoxType.YOLO:
            bboxes = torch.cat([xs_rel, ys_rel, w_rel, h_rel], dim=2)
        elif self.bbox_type == BBoxType.COCO:
            bboxes = torch.cat([xs_abs - w_abs/2, ys_abs - h_abs/2, w_abs, h_abs], dim=2)
        elif self.bbox_type == BBoxType.PASCAL_VOC:
            bboxes = torch.cat([xs_abs - w_abs/2, ys_abs - h_abs/2,
                                xs_abs + w_abs/2, ys_abs + h_abs/2], dim=2)
        elif self.bbox_type == BBoxType.ALBUMENTATIONS:
            bboxes = torch.cat([xs_rel - 0.5*w_rel, ys_rel - 0.5*h_rel, xs_rel + 0.5*w_rel, ys_rel + 0.5*h_rel], dim=2)
        else:
            raise RuntimeError(f'unsupported bbox type: {self.bbox_type}')

        detections = (bboxes.type(torch.LongTensor), scores, clses)

        return detections


    @staticmethod
    def pseudo_nms(fmap, pool_size=3):
        r"""
        apply max pooling to get the same effect of nms

        Args:
            fmap(Tensor): output tensor of previous step
            pool_size(int): size of max-pooling
        """
        pad = (pool_size - 1) // 2
        fmap_max = torch.nn.functional.max_pool2d(fmap, pool_size, stride=1, padding=pad)
        keep = (fmap_max == fmap).float()
        return fmap * keep

    @staticmethod
    def topk_score(scores, K=40):
        """
        get top K point in score map
        """
        batch, channel, height, width = scores.shape

        # get topk score and its index in every H x W(channel dim) feature map
        topk_scores, topk_inds = torch.topk(scores.reshape(batch, channel, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        # get all topk in in a batch
        topk_score, index = torch.topk(topk_scores.reshape(batch, -1), K)
        # div by K because index is grouped by K(C x K shape)
        topk_clses = (index / K).int()
        topk_inds = gather_feature(topk_inds.view(batch, -1, 1), index).reshape(batch, K)
        topk_ys = gather_feature(topk_ys.reshape(batch, -1, 1), index).reshape(batch, K)
        topk_xs = gather_feature(topk_xs.reshape(batch, -1, 1), index).reshape(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


class EncodeBBoxes(torch.nn.Module):

    def __init__(self,
                 bbox_type='yolo',
                 num_classes=2,
                 output_size=(224, 224),
                 min_overlap=0.7,
                 tensor_dim=0,
                 **kwargs):
        super(EncodeBBoxes, self).__init__()
        self.bbox_type = BBoxType(bbox_type)
        self.bbox_class = BBoxesClass[self.bbox_type]
        self.num_classes = num_classes
        self.output_size = output_size
        self.min_overlap = min_overlap
        self.tensor_dim = tensor_dim

    def forward(self, x):
        return x

    def generate(self, labels):

        maps = []
        for idx in range(len(labels)):
            # img_size = (data['height'], data['width'])

            # init gt tensors
            gt_scoremap = torch.zeros(self.num_classes, *self.output_size)
            gt_wh = torch.zeros(2, *self.output_size)

            classes = torch.tensor([l[4] for l in labels[idx]])
            boxes = self.bbox_class([l[0:4] for l in labels[idx]])
            if len(boxes) != 0:
                centers = boxes.get_centers()
                centers_int = (centers*torch.Tensor(self.output_size)).type(torch.LongTensor)
                wh_int = (boxes.wh*torch.Tensor(self.output_size)).type(torch.LongTensor)
                self.generate_score_map(
                    gt_scoremap, classes, wh_int,
                    centers_int, self.min_overlap,
                )
                for b in range(len(boxes)):
                    cy, cx = centers_int[b]
                    gt_wh[:, cy, cx] = boxes.wh[b, :]

            maps.append(torch.cat([gt_scoremap, gt_wh]))

        return torch.stack(maps, dim=0)

    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = EncodeBBoxes.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            EncodeBBoxes.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        box_tensor = torch.Tensor(box_size)
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = EncodeBBoxes.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
        # return fmap
