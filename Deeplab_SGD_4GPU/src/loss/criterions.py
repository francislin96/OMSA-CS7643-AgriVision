import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, args):
        super(DiceLoss, self).__init__()
        self.classes = args.num_classes
        self.ignore_index = args.ignore_index
        self.eps = 1e-6

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
    def __init__(self, args, alpha=0.5, beta=0.5):
        """
        Initialize TverskyLoss module.
        
        Args:
            alpha (float): Controls the penalty for false negatives.
            beta (float): Controls the penalty for false positives.
            smooth (float): Small constant to avoid division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.smooth = 1e-6

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
    def __init__(self, args):
        """
        Initialize FocalTverskyLoss module.

        Args:
            alpha (float): Controls the penalty for false negatives.
            beta (float): Controls the penalty for false positives.
            gamma (float): Focusing parameter that adjusts the rate at which easy examples are down-weighted.
            smooth (float): Small constant to avoid division by zero.
        """
        super(FocalTverskyLoss, self).__init__()
        self.args = args
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.smooth = 1e-6
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

### Adapted from https://github.com/samleoqh/MSCG-Net    
class ACW_loss(nn.Module):
    def __init__(self, args, ini_weight=0, ini_iteration=0, eps=1e-5):
        super(ACW_loss, self).__init__()
        self.ignore_index = args.ignore_index
        self.weight = ini_weight
        self.itr = ini_iteration
        self.eps = eps

    def forward(self, prediction, target):
        """
        pred :    shape (N, C, H, W)
        target :  shape (N, H, W) ground truth
        return:  loss_acw
        """
        pred = F.softmax(prediction, 1)
        one_hot_label, mask = self.encode_one_hot_label(pred, target)

        acw = self.adaptive_class_weight(pred, one_hot_label, mask)

        err = torch.pow((one_hot_label - pred), 2)
    
        pnc = err - ((1. - err + self.eps) / (1. + err + self.eps)).log()
        loss_pnc = torch.sum(acw * pnc, 1)


        intersection = 2 * torch.sum(pred * one_hot_label, dim=(0, 2, 3)) + self.eps
        union = pred + one_hot_label

        if mask is not None:
            union[mask] = 0

        union = torch.sum(union, dim=(0, 2, 3)) + self.eps
        dice = intersection / union

        return loss_pnc.mean() - dice.mean().log()

    def adaptive_class_weight(self, pred, one_hot_label, mask=None):
        self.itr += 1

        sum_class = torch.sum(one_hot_label, dim=(0, 2, 3))
        sum_norm = sum_class / sum_class.sum()

        self.weight = (self.weight * (self.itr - 1) + sum_norm) / self.itr
        mfb = self.weight.mean() / (self.weight + self.eps)
        mfb = mfb / mfb.sum()
        mfb = torch.clamp(mfb, min=0.001, max=1.0)
        acw = (1. + pred + one_hot_label) * mfb.unsqueeze(-1).unsqueeze(-1)

        if mask is not None:
            acw[mask] = 0

        return acw

    def encode_one_hot_label(self, pred, target):
        one_hot_label = pred.detach() * 0
        if self.ignore_index is not None:
            mask = (target == self.ignore_index)
            target = target.clone()
            target[mask] = 0
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(one_hot_label)
            one_hot_label[mask] = 0
            return one_hot_label, mask
        else:
            one_hot_label.scatter_(1, target.unsqueeze(1), 1)
            return one_hot_label, None
        
