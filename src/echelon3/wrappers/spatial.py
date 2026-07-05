import torch
import torch.nn.functional as F

from echelon3.wrappers.basic import BasicWrapper

class UpsampleDownsample(BasicWrapper):

    def __init__(self, core, smallest_size, **kwargs):
        super(UpsampleDownsample, self).__init__(core)

        self.smallest_size = smallest_size

    def forward(self, x):

        smallest_axis = torch.argmin(torch.tensor(x.shape[2:]))
        largest_axis = torch.argmax(torch.tensor(x.shape[2:]))

        if largest_axis == smallest_axis:
            largest_axis = (largest_axis + 1) % 2


        new_shape = torch.tensor([0, 0])

        new_shape[smallest_axis] = self.smallest_size
        new_shape[largest_axis] = int(x.shape[2+largest_axis] * (self.smallest_size / x.shape[2+smallest_axis]))

        x1 = F.adaptive_avg_pool2d(x, new_shape)

        x2 = self.core(x1)

        x3 = F.interpolate(input=x2, size=x.shape[2:],
                                     mode='bilinear', align_corners=True)

        return x3



class PadToMultiplier(BasicWrapper):

    def __init__(self, core, m, constant, **kwargs):
        super(PadToMultiplier, self).__init__(core)

        self.m = m
        self.constant = float(constant)

    def get_padded_addon(self, x: int):
        return int(((x + self.m - 1) // self.m) * self.m - x)

    def pad(self, x) -> tuple[torch.Tensor, tuple[int, int, int, int]]:
        padded_addons = [self.get_padded_addon(x.shape[i]) for i in [-2, -1]]
        paddings = (int(padded_addons[-2]//2), int(padded_addons[-2] - padded_addons[-2]//2),
                    int(padded_addons[-1]//2), int(padded_addons[-1] - padded_addons[-1]//2))

        return F.pad(x, paddings, 'constant', self.constant), paddings


    def revert(self, x, paddings: tuple[int, int, int, int]):
        return x[..., paddings[0]:(paddings[0]+x.shape[-2]), paddings[2]:(paddings[2]+x.shape[-1])]

    def forward(self, x):

        x, paddings = self.pad(x)

        x = self.core(x)
        x = self.revert(x, paddings=paddings)

        return x


