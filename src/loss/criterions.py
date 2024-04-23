import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, classes: int, ignore_index=None, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.classes = classes
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits, target):
        """
        logits : the output from the model (before softmax) with shape [N, C, H, W]
        target : the ground truth with shape [N, H, W]
        """
        # Apply softmax to the logitss
        logits = F.softmax(logits, dim=1)

        # Create the one-hot encoded version of target
        one_hot_target = torch.zeros(logits.size()).to(target.device)
        one_hot_target.scatter_(1, target.unsqueeze(1), 1)
        
        if self.ignore_index is not None:
            mask = target != self.ignore_index
            logits = logits * mask.unsqueeze(1)
            one_hot_target = one_hot_target * mask.unsqueeze(1)
        
        # Calculate Dice score per class
        intersection = torch.sum(logits * one_hot_target, dim=(0, 2, 3))
        union = torch.sum(logits, dim=(0, 2, 3)) + torch.sum(one_hot_target, dim=(0, 2, 3))

        dice_score = (2. * intersection + self.eps) / (union + self.eps)
        dice_loss = 1. - dice_score
        
        # Exclude the ignore_index from loss calculation
        if self.ignore_index is not None:
            dice_loss = dice_loss[one_hot_target.sum((0, 2, 3)) != 0]

        # Average the Dice loss across all classes
        return dice_loss.mean()
    
class TverskyLoss(nn.Module):
    def __init__(self, args, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Initialize TverskyLoss module.
        
        Args:
            alpha (float): Controls the penalty for false negatives.
            beta (float): Controls the penalty for false positives.
            smooth (float): Small constant to avoid division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.args = args
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        Calculate the Tversky loss between `logits` and `targets`
        
        Args:
            logits (torch.Tensor): A tensor of logits (before softmax) of shape (N, C, H, W), where N is the batch size,
                                    C is the number of classes, and H, W are the height and width of the output.
            targets (torch.Tensor): A tensor of shape (N, H, W) where each value is 
                                    an integer representing the class index.
        
        Returns:
            torch.Tensor: Scalar tensor containing the Tversky loss.
        """
        # Get the number of classes from the logits
        num_classes = logits.shape[1]

        # Convert targets to one hot encoding
        true_1_hot = torch.eye(num_classes).to(self.args.device)[targets].permute(0, 3, 1, 2)
        # true_1_hot = torch.eye(num_classes)[targets].permute(0, 3, 1, 2)
        true_1_hot = true_1_hot.type(logits.type())
        # true_1_hot = true_1_hot.to(self.args.device)

        # Apply softmax to logits to get probability distribution
        probs = torch.softmax(logits, dim=1)

        # Calculate true positives, false negatives, and false positives
        true_pos = torch.sum(probs * true_1_hot, dim=(2, 3))
        false_neg = torch.sum(true_1_hot * (1 - probs), dim=(2, 3))
        false_pos = torch.sum((1 - true_1_hot) * probs, dim=(2, 3))

        # Calculate the Tversky index for each class and then average over all classes
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        tversky_loss = 1 - tversky_index.mean()

        return tversky_loss
    
class FocalTverskyLoss(nn.Module):
    def __init__(self, args, alpha=0.5, beta=0.5, gamma=2.0, smooth=1e-6):
        """
        Initialize FocalTverskyLoss module.

        Args:
            alpha (float): Controls the penalty for false negatives.
            beta (float): Controls the penalty for false positives.
            gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
            smooth (float): Small constant to avoid division by zero.
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.args = args

    def forward(self, logits, targets):
        """
        Calculate the Focal Tversky loss between `logits` and `targets`
        
        Args:
            logits (torch.Tensor): A tensor of logits (before softmax) of shape (N, C, H, W), where N is the batch size,
                                    C is the number of classes, and H, W are the height and width of the output.
            targets (torch.Tensor): A tensor of shape (N, H, W) where each value is 
                                    an integer representing the class index.
        
        Returns:
            torch.Tensor: Scalar tensor containing the Focal Tversky loss.
        """
        # Get the number of classes from the logits
        num_classes = logits.shape[1]

        # Convert targets to one hot encoding
        true_1_hot = torch.eye(num_classes).to(self.args.device)[targets].permute(0, 3, 1, 2)
        true_1_hot = true_1_hot.type(logits.type())
        true_1_hot = true_1_hot.to(logits.device)

        # Apply softmax to logits to get probability distribution
        probs = torch.softmax(logits, dim=1)

        # Calculate true positives, false negatives, and false positives
        true_pos = torch.sum(probs * true_1_hot, dim=(2, 3))
        false_neg = torch.sum(true_1_hot * (1 - probs), dim=(2, 3))
        false_pos = torch.sum((1 - true_1_hot) * probs, dim=(2, 3))

        # Calculate the Tversky index for each class and then average over all classes
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        tversky_loss = 1 - tversky_index

        # Apply the focusing parameter
        focal_tversky_loss = torch.pow(tversky_loss, self.gamma).mean()

        return focal_tversky_loss