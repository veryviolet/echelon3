from abc import abstractmethod
import copy
from enum import Enum, unique
from typing import Iterator, List, Tuple, Union

import numpy as np
import torch

class BBoxType(Enum):
    YOLO = 'yolo'
    COCO = 'coco'
    PASCAL_VOC = 'pascal_voc'
    ALBUMENTATIONS = 'albumentations'


class BBoxes:

    data: torch.Tensor

    def __init__(self, data: Union[torch.Tensor, List[List[Union[int, float]]]]):
        self.data = torch.Tensor(data)

    def to(self, device: str) -> "BBoxes":
        return BBoxes(self.data.to(device))

    def clone(self) -> "BBoxes":
        return BBoxes(self.data.clone())

    @abstractmethod
    def convert(self, new_type: BBoxType, size: List[int] = None) -> "BBoxes":
        pass

    @abstractmethod
    def area(self, size: Union[Tuple[int], torch.Tensor] = None) -> torch.Tensor:
        pass

    @abstractmethod
    def clip(self, size: Union[List[int], torch.Tensor] = None):
        pass

    def nonempty(self, threshold: Union[int, float] = 0) -> torch.Tensor:
        return (self.wh[:, 0] > threshold) & (self.wh[:, 1] > threshold)

    def __repr__(self) -> str:
        return "BBoxes(" + str(self.data) + ")"

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BBoxes":
        if isinstance(item, int):
            return BBoxes(self.data[item].view(1, -1))
        b = self.data[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return BBoxes(b)

    def __len__(self) -> int:
        return self.data.shape[0]

    @abstractmethod
    def get_centers(self) -> torch.Tensor:
        pass

    @property
    def device(self) -> torch.device:
        return self.data.device

    def __iter__(self) -> Iterator[torch.Tensor]:
        yield from self.data

    @property
    @abstractmethod
    def lt(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def rb(self) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def wh(self) -> torch.Tensor:
        pass


class YoloBBoxes(BBoxes):

    def area(self, size: Union[Tuple[int], torch.Tensor] = None) -> torch.Tensor:
        return (size[0]*self.data[:, 2]).type(torch.LongTensor)*(size[1]*self.data[:, 3]).type(torch.LongTensor)

    def clip(self, size: Union[List[int], torch.Tensor] = None):
        self.data[:, 0].clamp_(min=0.0, max=1.0)
        self.data[:, 1].clamp_(min=0.0, max=1.0)
        self.data[:, 2].clamp_(min=0.0, max=torch.minimum(1.0 - self.data[:, 0], self.data[:, 0]))
        self.data[:, 3].clamp_(min=0.0, max=torch.minimum(1.0 - self.data[:, 1], self.data[:, 1]))

    def get_centers(self) -> torch.Tensor:
        return self.data[:, 0:2]

    @property
    def lt(self) -> torch.Tensor:
        return self.data[:, 0:2] - 0.5 * self.data[:, 2:]

    @property
    def rb(self) -> torch.Tensor:
        return self.data[:, 0:2] + 0.5 * self.data[:, 2:]

    @property
    def wh(self) -> torch.Tensor:
        return self.data[:, 2:]

    def convert(self, new_type: BBoxType, size: List[int] = None) -> "BBoxes":
        if new_type == BBoxType.YOLO:
            return self

        new_data = self.data.clone()

        if new_data.nelement() == 0:
            return BBoxesClass[new_type](new_data)

        size_tensor = list(size) if type(size) != list else size

        if new_type == BBoxType.COCO:
            new_data[:, 0] = (self.data[:, 0] - 0.5*self.data[:, 2])*size_tensor[0]
            new_data[:, 1] = (self.data[:, 1] - 0.5*self.data[:, 3])*size_tensor[1]
            new_data = new_data.type(torch.LongTensor)
            return CocoBBoxes(new_data)
        elif new_type == BBoxType.PASCAL_VOC:
            new_data[:, 0] = (self.data[:, 0] - 0.5*self.data[:, 2])*size_tensor[0]
            new_data[:, 1] = (self.data[:, 1] - 0.5*self.data[:, 3])*size_tensor[1]
            new_data[:, 2] = (self.data[:, 0] + 0.5*self.data[:, 2])*size_tensor[0]
            new_data[:, 3] = (self.data[:, 1] + 0.5*self.data[:, 3])*size_tensor[1]
            new_data = new_data.type(torch.LongTensor)
            return PascalVocBBoxes(new_data)
        elif new_type == BBoxType.ALBUMENTATIONS:
            new_data[:, 0] = self.data[:, 0] - 0.5*self.data[:, 2]
            new_data[:, 1] = self.data[:, 1] - 0.5*self.data[:, 3]
            new_data[:, 2] = self.data[:, 0] + 0.5*self.data[:, 2]
            new_data[:, 3] = self.data[:, 1] + 0.5*self.data[:, 3]
            return AlbumentationsBBoxes(new_data)


class CocoBBoxes(BBoxes):

    def convert(self, new_type: BBoxType, size: List[int] = None) -> "BBoxes":

        if new_type == BBoxType.COCO:
            return self

        new_data = self.data.clone()

        if new_data.nelement() == 0:
            return BBoxesClass[new_type](new_data)

        size_tensor = list(size) if type(size) != list else size

        if new_type == BBoxType.YOLO:
            new_data = new_data.type(torch.FloatTensor)
            new_data[:, 0] = (self.data[:, 0] + (self.data[:, 2] / 2)) / size_tensor[0]
            new_data[:, 1] = (self.data[:, 1] + (self.data[:, 3] / 2)) / size_tensor[1]
            new_data[:, 2] = self.data[:, 2] / size_tensor[0]
            new_data[:, 3] = self.data[:, 3] / size_tensor[1]
            return YoloBBoxes(new_data)
        elif new_type == BBoxType.PASCAL_VOC:
            new_data[:, 2] = self.data[:, 0] + self.data[:, 2] - 1
            new_data[:, 3] = self.data[:, 1] + self.data[:, 3] - 1
            return PascalVocBBoxes(new_data)
        elif new_type == BBoxType.ALBUMENTATIONS:
            new_data = new_data.type(torch.FloatTensor)
            new_data[:, 2] = (self.data[:, 0] + self.data[:, 2] - 1.0)/size_tensor[0]
            new_data[:, 3] = (self.data[:, 1] + self.data[:, 3] - 1.0)/size_tensor[1]
            return AlbumentationsBBoxes(new_data)


    def area(self, size: Union[Tuple[int], torch.Tensor] = None) -> torch.Tensor:
        return self.data[:, 2]*self.data[:, 3]

    def clip(self, size: Union[List[int], torch.Tensor] = None):
        self.data[:, 0].clamp_(min=0, max=size[0])
        self.data[:, 1].clamp_(min=0, max=size[1])
        self.data[:, 2].clamp_(min=0, max=size[0] - self.data[:, 0])
        self.data[:, 3].clamp_(min=0, max=size[1] - self.data[:, 1])

    def get_centers(self) -> torch.Tensor:
        return self.lt + (self.wh / 2).type(torch.LongTensor)

    @property
    def lt(self) -> torch.Tensor:
        return self.data[:, 0:2]

    @property
    def rb(self) -> torch.Tensor:
        return self.data[:, 0:2] + self.data[:, 2:]

    @property
    def wh(self) -> torch.Tensor:
        return self.data[:, 2:]


class PascalVocBBoxes(BBoxes):

    def convert(self, new_type: BBoxType, size: List[int] = None) -> "BBoxes":

        if new_type == BBoxType.PASCAL_VOC:
            return self

        new_data = self.data.clone()

        if new_data.nelement() == 0:
            return BBoxesClass[new_type](new_data)

        size_tensor = list(size) if type(size) != list else size

        if new_type == BBoxType.YOLO:
            new_data = new_data.type(torch.FloatTensor)
            new_data[:, 0] = ((self.data[:, 0] + self.data[:, 2]) / 2) / size_tensor[0]
            new_data[:, 1] = ((self.data[:, 1] + self.data[:, 3]) / 2) / size_tensor[1]
            new_data[:, 2] = (self.data[:, 2] - self.data[:, 0] + 1.0) / size_tensor[0]
            new_data[:, 3] = (self.data[:, 3] - self.data[:, 1] + 1.0) / size_tensor[1]
            return YoloBBoxes(new_data)
        elif new_type == BBoxType.COCO:
            new_data[:, 2] = self.data[:, 2] - self.data[:, 0] + 1
            new_data[:, 3] = self.data[:, 3] - self.data[:, 1] + 1
            return CocoBBoxes(new_data)
        elif new_type == BBoxType.ALBUMENTATIONS:
            new_data = new_data.type(torch.FloatTensor)
            new_data[:, 0] = self.data[:, 0]/size_tensor[0]
            new_data[:, 1] = self.data[:, 1]/size_tensor[1]
            new_data[:, 2] = self.data[:, 2]/size_tensor[0]
            new_data[:, 3] = self.data[:, 3]/size_tensor[1]
            return AlbumentationsBBoxes(new_data)

    def area(self, size: Union[Tuple[int], torch.Tensor] = None) -> torch.Tensor:
        return (self.data[:, 2] - self.data[:, 0]) * (self.data[:, 3] - self.data[:, 1])

    def clip(self, size: Union[List[int], torch.Tensor] = None):
        self.data[:, 0].clamp_(min=0, max=size[0])
        self.data[:, 1].clamp_(min=0, max=size[1])
        self.data[:, 2].clamp_(min=0, max=size[0])
        self.data[:, 3].clamp_(min=0, max=size[1])

    def get_centers(self) -> torch.Tensor:
        return ((self.lt + self.rb) / 2).type(torch.LongTensor)

    @property
    def lt(self) -> torch.Tensor:
        return self.data[:, 0:2]

    @property
    def rb(self) -> torch.Tensor:
        return self.data[:, 2:]

    @property
    def wh(self) -> torch.Tensor:
        return self.data[:, 2:] - self.data[:, 0:2] + 1


class AlbumentationsBBoxes(BBoxes):

    def convert(self, new_type: BBoxType, size: List[int] = None) -> "BBoxes":

        if new_type == BBoxType.ALBUMENTATIONS:
            return self

        new_data = self.data.clone()

        if new_data.nelement() == 0:
            return BBoxesClass[new_type](new_data)

        size_tensor = list(size) if type(size) != list else size

        if new_type == BBoxType.YOLO:
            new_data[:, 0] = 0.5*(self.data[:, 0] + self.data[:, 2])
            new_data[:, 1] = 0.5*(self.data[:, 1] + self.data[:, 3])
            new_data[:, 2] = self.data[:, 2] - self.data[:, 0]
            new_data[:, 3] = self.data[:, 3] - self.data[:, 1]
            return YoloBBoxes(new_data)
        elif new_type == BBoxType.COCO:
            new_data[:, 0] = self.data[:, 0]*size_tensor[0]
            new_data[:, 1] = self.data[:, 1]*size_tensor[1]
            new_data[:, 2] = (self.data[:, 2] - self.data[:, 0])*size_tensor[0]
            new_data[:, 3] = (self.data[:, 3] - self.data[:, 1])*size_tensor[1]
            new_data = new_data.type(torch.LongTensor)
            return CocoBBoxes(new_data)
        elif new_type == BBoxType.PASCAL_VOC:
            new_data[:, 0] = self.data[:, 0]*size_tensor[0]
            new_data[:, 1] = self.data[:, 1]*size_tensor[1]
            new_data[:, 2] = self.data[:, 2]*size_tensor[0]
            new_data[:, 3] = self.data[:, 3]*size_tensor[1]
            new_data = new_data.type(torch.LongTensor)
            return PascalVocBBoxes(new_data)

    def area(self, size: Union[Tuple[int], torch.Tensor] = None) -> torch.Tensor:
        return (size[0]*(self.data[:, 2] - self.data[:, 0])).type(torch.LongTensor) *\
            (size[1]*(self.data[:, 3] - self.data[:, 1])).type(torch.LongTensor)

    def clip(self, size: Union[List[int], torch.Tensor] = None):
        self.data[:, 0].clamp_(min=0.0, max=1.0)
        self.data[:, 1].clamp_(min=0.0, max=1.0)
        self.data[:, 2].clamp_(min=0.0, max=1.0)
        self.data[:, 3].clamp_(min=0.0, max=1.0)

    def get_centers(self) -> torch.Tensor:
        return 0.5 * (self.lt + self.rb)

    @property
    def lt(self) -> torch.Tensor:
        return self.data[:, 0:2]

    @property
    def rb(self) -> torch.Tensor:
        return self.data[:, 2:]

    @property
    def wh(self) -> torch.Tensor:
        return self.rb - self.lt


BBoxesClass = {
    BBoxType.YOLO: YoloBBoxes,
    BBoxType.COCO: CocoBBoxes,
    BBoxType.PASCAL_VOC: PascalVocBBoxes,
    BBoxType.ALBUMENTATIONS: AlbumentationsBBoxes
}


def pairwise_iou(boxes1: BBoxes, boxes2: BBoxes) -> torch.Tensor:
    """
    Given two lists of boxes of size N and M,
    compute the IoU (intersection over union)
    between __all__ N x M pairs of boxes.
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
        boxes1,boxes2 (Boxes): two `Boxes`. Contains N & M boxes, respectively.

    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = boxes1.area()
    area2 = boxes2.area()

    width_height = torch.min(boxes1.wh[:, None, :], boxes2.wh) - torch.max(boxes1.wh[:, None, :], boxes2.wh)  # [N,M,2]

    width_height.clamp_(min=0)  # [N,M,2]
    inter = width_height.prod(dim=2)  # [N,M]
    del width_height

    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / (area1[:, None] + area2 - inter),
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou


def matched_boxlist_iou(boxes1: BBoxes, boxes2: BBoxes) -> torch.Tensor:
    """
    Compute pairwise intersection over union (IOU) of two sets of matched
    boxes. The box order must be (xmin, ymin, xmax, ymax).
    Similar to boxlist_iou, but computes only diagonal elements of the matrix
    Arguments:
        boxes1: (Boxes) bounding boxes, sized [N,4].
        boxes2: (Boxes) bounding boxes, sized [N,4].
    Returns:
        (tensor) iou, sized [N].
    """
    assert len(boxes1) == len(boxes2), (
        "boxlists should have the same"
        "number of entries, got {}, {}".format(len(boxes1), len(boxes2))
    )
    area1 = boxes1.area()  # [N]
    area2 = boxes2.area()  # [N]
    box1, box2 = boxes1.tensor, boxes2.tensor
    lt = torch.max(boxes1.lt, boxes2.lt)  # [N,2]
    rb = torch.min(boxes1.rb, boxes2.rb)  # [N,2]
    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    iou = inter / (area1 + area2 - inter)  # [N]
    return iou


def coords_to_relative(coords, size: Union[Tuple[int], torch.Tensor]):
    pass

def coords_to_absolute(coords, size: Union[Tuple[int], torch.Tensor]):
    if len(coords.shape) == 0 or coords.shape[0] == 0:
        return coords
    if len(coords.shape) == 1:
        coords = torch.unsqueeze(coords, dim=0)

    if torch.is_floating_point(coords):
        coords[:, 0] *= size[1]
        coords[:, 2] *= size[1]
        coords[:, 1] *= size[0]
        coords[:, 3] *= size[0]
        coords = coords.type(torch.LongTensor)

    return coords


