import torch

def miou(pred, target, num_classes, eps=1e-6):
    """
    Compute the soft mean Intersection over Union (mIoU) between predicted and target segmentation masks.

    Args:
        pred (torch.Tensor): Predicted logits of shape (N, C), or (..., C), where C is number of classes.
        target (torch.Tensor): Ground truth one-hot encoding of shape (N, C) or (..., C).
        num_classes (int): Number of classes in the segmentation task.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        float: The mean soft IoU score across all classes.
    """

    # Apply softmax to logits along the last dimension to get predicted probabilities
    pred_probs = torch.softmax(pred, dim=-1)
    
    # Reshape for convenience: both as (..., C)
    if pred_probs.dim() == 2:
        pred_flat = pred_probs
        target_flat = target
    else:
        pred_flat = pred_probs.view(-1, num_classes)
        target_flat = target.view(-1, num_classes)

    ious = []
    for c in range(num_classes):
        pred_c = pred_flat[:, c]
        target_c = target_flat[:, c]
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        iou = (intersection + eps) / (union + eps)
        ious.append(iou)

    miou = torch.mean(torch.stack(ious)).item()
    return miou