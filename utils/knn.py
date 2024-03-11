import torch


def knn_gather(x, idx):
    """
    :param x: (B, N, C)
    :param idx: (B, N, K)
    :return: (B, N, K, C)
    """
    C = x.shape[-1]
    B, N, K = idx.shape
    idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, C)
    x_out = x[:, :, None].expand(-1, -1, K, -1).gather(1, idx_expanded)

    return x_out
