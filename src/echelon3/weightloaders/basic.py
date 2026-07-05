import torch

class WeightsLoader:

    def load(self, net, weights, device):
        net.load_state_dict(torch.load(weights, map_location=device))
