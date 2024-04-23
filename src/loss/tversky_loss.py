import torch
import torch.nn as nn

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

    def forward(self, outputs, targets):
        """
        Calculate the Tversky loss between `outputs` and `targets`
        
        Args:
            outputs (torch.Tensor): A tensor of logits (before softmax) of shape (N, C, H, W), where N is the batch size,
                                    C is the number of classes, and H, W are the height and width of the output.
            targets (torch.Tensor): A tensor of shape (N, H, W) where each value is 
                                    an integer representing the class index.
        
        Returns:
            torch.Tensor: Scalar tensor containing the Tversky loss.
        """
        # Get the number of classes from the outputs
        num_classes = outputs.shape[1]

        # Convert targets to one hot encoding
        true_1_hot = torch.eye(num_classes).to(self.args.device)[targets].permute(0, 3, 1, 2)
        # true_1_hot = torch.eye(num_classes)[targets].permute(0, 3, 1, 2)
        true_1_hot = true_1_hot.type(outputs.type())
        # true_1_hot = true_1_hot.to(self.args.device)

        # Apply softmax to outputs to get probability distribution
        probs = torch.softmax(outputs, dim=1)

        # Calculate true positives, false negatives, and false positives
        true_pos = torch.sum(probs * true_1_hot, dim=(2, 3))
        false_neg = torch.sum(true_1_hot * (1 - probs), dim=(2, 3))
        false_pos = torch.sum((1 - true_1_hot) * probs, dim=(2, 3))

        # Calculate the Tversky index for each class and then average over all classes
        tversky_index = (true_pos + self.smooth) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.smooth)
        tversky_loss = 1 - tversky_index.mean()

        return tversky_loss