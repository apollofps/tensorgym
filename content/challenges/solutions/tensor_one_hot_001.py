import torch


def one_hot_encode(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.long()
    result = torch.zeros(labels.shape[0], num_classes)
    result.scatter_(1, labels.unsqueeze(1), 1.0)
    return result
