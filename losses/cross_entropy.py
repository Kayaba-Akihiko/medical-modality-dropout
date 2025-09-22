#  Copyright (c) 2024.
#  by Yi GU <yi.gu@sinicx.com>,
#  OMRON SINIC X.
#  All rights reserved.
#  This file can not be copied and/or distributed
#  without the express permission of Yi GU.

import torch
from typing import Optional, override
from torch import Tensor
from torch.nn import BCEWithLogitsLoss as _BCEWithLogitsLoss
import torch.nn.functional as F
from einops import rearrange


class BCEWithLogitsLoss(_BCEWithLogitsLoss):

    @override
    def __init__(
            self,
            weight: Optional[Tensor] = None,
            size_average=None,
            reduce=None,
            reduction: str = "mean",
            pos_weight: Optional[Tensor] = None,
            label_smoothing: Optional[float] = None,
            is_multi_label: bool = False,
    ) -> None:
        super().__init__(
            weight=weight,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
            pos_weight=pos_weight,
        )

        if label_smoothing is not None and label_smoothing > 0.:
            self.register_buffer(
                'label_smoothing',
                torch.tensor(label_smoothing),
            )
            self.register_buffer(
                'confidence', torch.tensor(1. - label_smoothing))
        else:
            self.label_smoothing: Optional[Tensor] = None
            self.confidence: Optional[Tensor] = None
        self.is_multi_label = is_multi_label

    def forward(
            self, input: Tensor, target: Tensor) -> Tensor:

        with torch.no_grad():
            # (B, ...) LongTensor
            # (B, C, ...) FloatTensor
            logit_dim = input.shape[1]
            if input.ndim == target.ndim:
                # FloatTensor (B, C, ...)
                assert target.shape[1] == logit_dim
            elif input.ndim > target.ndim:
                # LongTensor (B, ...)
                assert not self.is_multi_label
                assert logit_dim > 1
                # (B, ...) -> (B, C, ...)
                target = F.one_hot(target, num_classes=logit_dim)
                target = rearrange(
                    target, 'b ... c ->b c ...')
                target.to(input.dtype)

        if self.label_smoothing is not None:
            return torch.compiler.disable(
                super().forward, recursive=True)(input, target)
        with torch.no_grad():
            self.label_smoothing: Tensor
            if self.is_multi_label or logit_dim == 1:
                label_smooth = self.label_smoothing
            else:
                assert logit_dim > 1
                label_smooth = self.label_smoothing / (logit_dim - 1)
            true_dist = (
                    target * self.confidence
                    + (1 - target) * label_smooth
            )

        return torch.compiler.disable(
            super().forward, recursive=True)(input, true_dist)
