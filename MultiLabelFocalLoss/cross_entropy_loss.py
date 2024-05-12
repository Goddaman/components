# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss


@MODELS.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probability distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls


@MODELS.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


@MODELS.register_module()
class CBFocalLoss(BaseWeightedLoss):
    """Class Balanced Focal Loss. Adapted from https://github.com/abhinanda-
    punnakkal/BABEL/. This loss is used in the skeleton-based action
    recognition baseline for BABEL.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        samples_per_cls (list[int]): The number of samples per class.
            Defaults to [].
        beta (float): Hyperparameter that controls the per class loss weight.
            Defaults to 0.9999.
        gamma (float): Hyperparameter of the focal loss. Defaults to 2.0.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 samples_per_cls: List[int] = [],
                 beta: float = 0.9999,
                 gamma: float = 2.) -> None:
        super().__init__(loss_weight=loss_weight)
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.weights = weights
        self.num_classes = len(weights)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        weights = torch.tensor(self.weights).float().to(cls_score.device)
        label_one_hot = F.one_hot(label, self.num_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(label_one_hot.shape[0], 1) * label_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)

        BCELoss = F.binary_cross_entropy_with_logits(
            input=cls_score, target=label_one_hot, reduction='none')

        modulator = 1.0
        if self.gamma:
            modulator = torch.exp(-self.gamma * label_one_hot * cls_score -
                                  self.gamma *
                                  torch.log(1 + torch.exp(-1.0 * cls_score)))

        loss = modulator * BCELoss
        weighted_loss = weights * loss

        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(label_one_hot)

        return focal_loss
    
@MODELS.register_module()
class MultiLabelFocalLoss(BaseWeightedLoss):
    """Multi-Label Focal Loss for action recognition.

    This loss is adapted for multi-label scenarios where each sample can belong to multiple classes.
    The loss applies Focal Loss to each class label independently.

    Args:
        gamma (float, optional): Focusing parameter to adjust the rate at which easy examples are down- weighted.
            Defaults to 2.0.
        alpha (float, optional): Balancing parameter to balance the importance of positive/negative examples.
            `alpha` for positive, `1-alpha` for negative.
            Defaults to 0.25.
        reduction (str, optional): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
            'none': no reduction will be applied.
            'mean': the sum of the output will be divided by the number of elements in the output.
            'sum': the output will be summed. Defaults to 'mean'.
        loss_weight (float, optional): Weight of the loss. Defaults to 1.0.
    """

    def __init__(self,
                 gamma: float = 2.0,
                 alpha: float = 0.25,
                 reduction: str = 'mean',
                 loss_weight: float = 1.0) -> None:
        super().__init__(loss_weight=loss_weight)
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Forward function to compute the multi-label focal loss.

        Args:
            cls_score (torch.Tensor): Predicted class scores, expected shape [batch_size, num_classes].
            label (torch.Tensor): Ground truth labels, same shape as `cls_score`, values should be 0 or   1.

        Returns:
            torch.Tensor: Computed multi-label focal loss.
        """
        # Ensure the cls_score is in proper shape
        if not (cls_score.dim() == 2 and cls_score.size() == label.size()):
            raise ValueError("cls_score and label must have the same shape [batch_size, num_classes].")

        # Compute the binary cross entropy loss without reduction
        BCE_loss = F.binary_cross_entropy_with_logits(cls_score, label, reduction='none')

        # Apply the focusing parameter
        pt = torch.exp(-BCE_loss)  # pt is the probability of being classified as the correct class
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss
