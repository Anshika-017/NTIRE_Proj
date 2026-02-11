import torch

def compute_mask(mask):
    return mask > 0

def rmse(pred, gt, mask):
    valid = compute_mask(mask)
    return torch.sqrt(((pred[valid] - gt[valid]) ** 2).mean())

def mae(pred, gt, mask):
    valid = compute_mask(mask)
    return torch.abs(pred[valid] - gt[valid]).mean()
