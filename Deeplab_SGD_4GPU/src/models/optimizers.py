from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR, LinearLR, ReduceLROnPlateau, LambdaLR
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict
import math

# --- Optimization & Scheduler---
def get_SGD(parameters, lr, momentum, weight_decay, nesterov):
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
    # optim_params = get_wd_param_list(model)
    return SGD(
        parameters,
        lr,
        momentum,
        weight_decay,
        nesterov
    )

def get_adam(parameters, lr, weight_decay):

    return Adam(
        params=parameters, lr=lr, weight_decay=weight_decay
    )


# def get_scheduler(args, optimizer):
#     """
#     Initialize and return scheduler object. FixMatch uses a learning rate scheduler, which applies a cosine learning
#     rate decay over the course of the training process.

#     Parameters
#     ----------
#     args: argparse.Namespace
#         Namespace that contains all command line arguments with their corresponding values
#     optimizer: torch.optim.Optimizer
#         Optimizer which is used for model training and for which learning rate is updated using the scheduler.
#     Returns
#     -------
#     scheduler: torch.optim.lr_scheduler.LambdaLR
#         Returns LambdaLR scheduler instance using a cosine learning rate decay.
#     """
#     return LambdaLR(
#         optimizer, lambda x: cosine_lr_decay(x, args.iters_per_epoch * args.epochs)
#     )

def get_exp_scheduler(optimizer, gamma):

    scheduler = ExponentialLR(optimizer, gamma=gamma)
    return scheduler

def init_params_lr(net, opt):
    bias_params = []
    nonbias_params = []
    for key, value in dict(net.named_parameters()).items():
        if value.requires_grad:
            if 'bias' in key:
                bias_params.append(value)
            else:
                nonbias_params.append(value)
    params = [
        {'params': nonbias_params,
         'lr': opt.lr,
         'weight_decay': opt.weight_decay},
        {'params': bias_params,
         'lr': opt.lr * 2.0,
         'weight_decay': 0}
    ]
    return params

### Adapted from https://github.com/samleoqh/MSCG-Net
class Lookahead(Optimizer):
    def __init__(self, base_optimizer, alpha=0.5, k=6):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        defaults = dict(lookahead_alpha=alpha, lookahead_k=k, lookahead_step=0)
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = base_optimizer.defaults
        self.defaults.update(defaults)
        self.state = defaultdict(dict)
        # manually add our defaults to the param groups
        for name, default in defaults.items():
            for group in self.param_groups:
                group.setdefault(name, default)

    def update_slow(self, group):
        for fast_p in group["params"]:
            if fast_p.grad is None:
                continue
            param_state = self.state[fast_p]
            if 'slow_buffer' not in param_state:
                param_state['slow_buffer'] = torch.empty_like(fast_p.data)
                param_state['slow_buffer'].copy_(fast_p.data)
            slow = param_state['slow_buffer']
            slow.add_(group['lookahead_alpha'], fast_p.data - slow)
            fast_p.data.copy_(slow)

    def sync_lookahead(self):
        for group in self.param_groups:
            self.update_slow(group)

    def step(self, closure=None):
        # print(self.k)
        #assert id(self.param_groups) == id(self.base_optimizer.param_groups)
        loss = self.base_optimizer.step(closure)
        for group in self.param_groups:
            group['lookahead_step'] += 1
            if group['lookahead_step'] % group['lookahead_k'] == 0:
                self.update_slow(group)
        return loss

    def state_dict(self):
        fast_state_dict = self.base_optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict['state']
        param_groups = fast_state_dict['param_groups']
        return {
            'state': fast_state,
            'slow_state': slow_state,
            'param_groups': param_groups,
        }

    def load_state_dict(self, state_dict):
        fast_state_dict = {
            'state': state_dict['state'],
            'param_groups': state_dict['param_groups'],
        }
        self.base_optimizer.load_state_dict(fast_state_dict)

        # We want to restore the slow state, but share param_groups reference
        # with base_optimizer. This is a bit redundant but least code
        slow_state_new = False
        if 'slow_state' not in state_dict:
            print('Loading state_dict from optimizer without Lookahead applied.')
            state_dict['slow_state'] = defaultdict(dict)
            slow_state_new = True
        slow_state_dict = {
            'state': state_dict['slow_state'],
            'param_groups': state_dict['param_groups'],  # this is pointless but saves code
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.param_groups = self.base_optimizer.param_groups  # make both ref same container
        if slow_state_new:
            # reapply defaults to catch missing lookahead specific ones
            for name, default in self.defaults.items():
                for group in self.param_groups:
                    group.setdefault(name, default)