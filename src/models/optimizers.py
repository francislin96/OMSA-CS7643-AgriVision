from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ReduceLROnPlateau, LambdaLR

from src.utils.training import get_wd_param_list, cosine_lr_decay

# --- Optimization & Scheduler---
def get_SGD(args, model):
    """
    Initialize and return SGD optimizer. The weight decay is applied to all parameters except for BatchNorm parameters,
    which are filtered out by the function get_wd_param_list.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    model: torch.nn.Module
        torch module, i.e. neural net, which is trained using FixMatch
    Returns
    -------
    optim: torch.optim.Optimizer
        Returns SGD optimizer which is used for model training
    """
    optim_params = get_wd_param_list(model)
    return SGD(
        optim_params,
        lr=args.lr,
        momentum=args.beta,
        weight_decay=args.wd,
        nesterov=True,
    )


def get_scheduler(args, optimizer):
    """
    Initialize and return scheduler object. FixMatch uses a learning rate scheduler, which applies a cosine learning
    rate decay over the course of the training process.

    Parameters
    ----------
    args: argparse.Namespace
        Namespace that contains all command line arguments with their corresponding values
    optimizer: torch.optim.Optimizer
        Optimizer which is used for model training and for which learning rate is updated using the scheduler.
    Returns
    -------
    scheduler: torch.optim.lr_scheduler.LambdaLR
        Returns LambdaLR scheduler instance using a cosine learning rate decay.
    """
    return LambdaLR(
        optimizer, lambda x: cosine_lr_decay(x, args.iters_per_epoch * args.epochs)
    )

def get_exp_scheduler():

    scheduler = ExponentialLR(optimizer, gamma=gamma)
    return scheduler