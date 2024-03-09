import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from .utils import compute_centroids, extract_features
from .method import FSmethod


class SimpleShot(FSmethod):
    '''
    Simple Shot method
    '''

    def __init__(self, args: argparse.Namespace):
        self.normalize = args.normalize
        super().__init__(args)

    def forward(self,
                x_s: torch.tensor,
                x_q: torch.tensor,
                y_s: torch.tensor,
                y_q: torch.tensor,
                model: nn.Module):
        """
        inputs:
            x_s : torch.Tensor of shape [s_shot, c, h, w]
            x_q : torch.Tensor of shape [q_shot, c, h, w]
            y_s : torch.Tensor of shape [s_shot]
            y_q : torch.Tensor of shape [q_shot]
        """
        if not self.training:
            with torch.no_grad():
                z_s = extract_features(x_s, model)
                z_q = extract_features(x_q, model)
        else:
            z_s = extract_features(x_s, model)
            z_q = extract_features(x_q, model)
        if self.normalize:
            z_s = F.normalize(z_s, dim=2)
            z_q = F.normalize(z_q, dim=2)
        centroids = compute_centroids(z_s, y_s)  # [batch, num_class, d]

        l2_distance = (- 2 * z_q.matmul(centroids.transpose(1, 2)) \
                        + (centroids**2).sum(2).unsqueeze(1)  # noqa: E127
                        + (z_q**2).sum(2).unsqueeze(-1))  # [batch, q_shot, num_class]

        preds_q = (-l2_distance).detach().softmax(-1).argmax(2)
        return None, preds_q
