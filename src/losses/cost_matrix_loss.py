import torch
from torch import nn
import torch.nn.functional as nnf

from src.torch_device import get_torch_device

# Inspired by https://github.com/agaldran/cost_sensitive_loss_classification/blob/master/utils/losses.py#L122
class CostMatrixLoss(nn.Module):

    def __init__(self, cost_matrix: torch.Tensor, is_gain_matrix: bool):
        super().__init__()

        if is_gain_matrix:
            cost_matrix = cost_matrix * -1

        self.cost_matrix = (cost_matrix / cost_matrix.abs().max()).to(get_torch_device())
        self.num_classes = cost_matrix.shape[0]

        self.normalization = nn.Softmax(dim=-1)


    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        predictions = self.normalization(predictions)

        prediction_loss = self._compute_loss_with_cost_matrix(predictions, targets)
        optimal_loss = self._compute_loss_with_cost_matrix(
            nnf.one_hot(targets, num_classes=self.num_classes),
            targets
        )

        return (prediction_loss - optimal_loss).mean()

    def _compute_loss_with_cost_matrix(self, predictions: torch.Tensor, targets: torch.Tensor):
        return (self.cost_matrix[targets, :] * predictions).sum(axis=-1)
