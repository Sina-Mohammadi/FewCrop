import argparse
import numpy as np
import torch
import torch.nn as nn
from .utils import rand_bbox


class _Loss(nn.Module):

    def __init__(self,
                 args: argparse.Namespace,
                 num_classes: int,
                 reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        self.reduction = reduction
        self.augmentation = None if args.augmentation not in ['mixup', 'cutmix'] \
                            else eval(f'self.{args.augmentation}')  # noqa: E127
        assert 0 <= args.label_smoothing < 1
        self.label_smoothing = args.label_smoothing
        self.num_classes = num_classes
        self.beta = args.beta
        self.cutmix = args.cutmix_prob

    def smooth_one_hot(self,
                       targets: torch.tensor):
        with torch.no_grad():
            new_targets = torch.empty(size=(targets.size(0), self.num_classes), device=targets.device)
            new_targets.fill_(self.label_smoothing / (self.num_classes-1))
            new_targets.scatter_(1, targets.unsqueeze(1), 1. - self.label_smoothing)
        return new_targets

    def mixup(self,
              input_: torch.tensor,
              one_hot_targets: torch.tensor,
              model: nn.Module):
        # Forward pass
        device = one_hot_targets.device

        # generate mixed sample and targets
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input_.size()[0]).to(device)
        target_a = one_hot_targets
        target_b = one_hot_targets[rand_index]
        mixed_input_ = lam * input_ + (1 - lam) * input_[rand_index]

        output = model(mixed_input_)
        loss = self.loss_fn(output, target_a) * lam + self.loss_fn(output, target_b) * (1. - lam)
        return loss

    def cutmix(self,
               input_: torch.tensor,
               one_hot_targets: torch.tensor,
               model: nn.Module):
        # generate mixed sample
        lam = np.random.beta(self.beta, self.beta)
        rand_index = torch.randperm(input_.size()[0]).cuda()
        target_a = one_hot_targets
        target_b = one_hot_targets[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(input_.size(), lam)
        input_[:, :, bbx1:bbx2, bby1:bby2] = input_[rand_index, :, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input_.size()[-1] * input_.size()[-2]))
        output = model(input_)
        loss = self.loss_fn(output, target_a) * lam + self.loss_fn(output, target_b) * (1. - lam)
        return loss

    def loss_fn(self,
                logits: torch.tensor,
                one_hot_targets: torch.tensor):
        raise NotImplementedError

    def forward(self,
                logits: torch.tensor,
                targets: torch.tensor,
                model: torch.nn.Module):
        raise NotImplementedError