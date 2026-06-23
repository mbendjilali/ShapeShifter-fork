import torch

def miou(pred, target, num_classes):
    """
    Compute the mean Intersection over Union (mIoU) between predicted and target segmentation masks.

    Args:
        pred (torch.Tensor): Predicted segmentation mask of shape (N, H, W) where N is the batch size.
        target (torch.Tensor): Ground truth segmentation mask of shape (N, H, W).
        num_classes (int): Number of classes in the segmentation task.

    Returns:
        float: The mean IoU score across all classes.
    """

    ious = []

    for c in range(num_classes):
        # Create binary masks for the current class
        pred_mask = (pred == c)
        target_mask = (target == c)

        # Calculate intersection and union
        intersection = (pred_mask & target_mask).sum().float()
        union = (pred_mask | target_mask).sum().float()

        if union > 0:
            ious.append(intersection / union)
    
    # Compute mean IoU
    miou = torch.mean(torch.tensor(ious)).item()
    
    return miou