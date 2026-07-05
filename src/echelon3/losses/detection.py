import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.focal_loss import sigmoid_focal_loss

from echelon3.utils.bbox_encode_decode import EncodeBBoxes

class DetectionFocalLoss(nn.Module):

    def __init__(self, alpha=2.0, beta=4.0):
        super(DetectionFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, prediction: Tensor, gt) -> Tensor:
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1.0 - gt, self.beta)
        # clamp min value is set to 1e-12 to maintain the numerical stability
        pred = torch.clamp(prediction, 1e-12)

        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1.0 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss


class WidthHeightLoss(nn.Module):

    def __init__(self):
        super(WidthHeightLoss, self).__init__()

    def forward(self, prediction: Tensor, gt) -> Tensor:
        pos_inds = gt.gt(0.0)
        if torch.sum(pos_inds) == 0:
            return torch.as_tensor(0.0)
        gt_whs = gt[pos_inds]
        pred_whs = prediction[pos_inds]
        loss = torch.nn.functional.l1_loss(pred_whs, gt_whs, reduction='sum')
        loss = loss / (pos_inds.sum() + 1e-4)
        return loss


class HeatmapBasedDetectionLoss(nn.Module):

    def __init__(self, *args, bbox_type='yolo', alpha=1.0, beta=4.0, output_size=(224, 224), num_classes=2, **kwargs):
        super(HeatmapBasedDetectionLoss, self).__init__(*args, **kwargs)
        self.encoder = EncodeBBoxes(bbox_type=bbox_type, output_size=output_size, num_classes=num_classes)
        self.num_classes = num_classes
        self.focal_loss = DetectionFocalLoss(alpha=alpha, beta=beta)
        self.sizes_loss = WidthHeightLoss()

    def forward(self, inputs: Tensor, target: list) -> Tensor:
        if type(inputs) != torch.Tensor:
            preds = inputs['heatmaps']
        else:
            preds = inputs
        heatmaps = self.encoder.generate(target).to(preds.device)
#        score_loss = torch.nn.functional.mse_loss(preds, heatmaps)
        #score_loss = torch.nn.functional.mse_loss(preds[:, :-2, :, :], heatmaps[:, :-2, :, :])
        sizes_loss = self.sizes_loss(preds[:, -2:, :, :], heatmaps[:, -2:, :, :])
#        focal = self.focal_loss(preds[:, 0:self.num_classes, :, :], heatmaps[:, 0:self.num_classes, :, :])
        focal_loss = sigmoid_focal_loss(preds[:, 0:self.num_classes, :, :], heatmaps[:, 0:self.num_classes, :, :],
                                        reduction='mean')
        return focal_loss + sizes_loss



