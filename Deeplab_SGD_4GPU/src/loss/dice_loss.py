import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, classes: int, ignore_index=None, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, prediction, target):
        """
        prediction : the output from the model (before softmax) with shape [N, C, H, W]
        target : the ground truth with shape [N, H, W]
        """
        # Apply softmax to the predictions
        prediction = F.softmax(prediction, dim=1)

        # Create the one-hot encoded version of target
        one_hot_target = torch.zeros(prediction.size()).to(target.device)
        one_hot_target.scatter_(1, target.unsqueeze(1), 1)
        
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            prediction = prediction * mask.unsqueeze(1)
            one_hot_target = one_hot_target * mask.unsqueeze(1)
        
        # Calculate Dice score per class
        intersection = torch.sum(prediction * one_hot_target, dim=(0, 2, 3))
        union = torch.sum(prediction, dim=(0, 2, 3)) + torch.sum(one_hot_target, dim=(0, 2, 3))

        dice_score = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1. - dice_score
        
        # Exclude the ignore_index from loss calculation
        if self.ignore_index is not None:
            dice_loss = dice_loss[one_hot_target.sum((0, 2, 3)) != 0]

        # Average the Dice loss across all classes
        return dice_loss.mean()