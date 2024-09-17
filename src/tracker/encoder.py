import sys
import warnings

import torch
import torch.nn.functional as F
import torchvision


class Precomputed:
    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self, x):
        features = x["annotations"][:, 11:]
        return F.normalize(features, p=2, dim=1)


def create_encoder(cfg, device):
    print(cfg)
    if cfg.name == "precomputed":
        return Precomputed(cfg)
    else:
        raise ValueError(f"Encoder {cfg.name} not found.")
