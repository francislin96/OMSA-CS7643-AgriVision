import torch
import torch.nn.functional as F

from src.models.create_models import deeplabv3_plus
from src.utils.training import add_weight_decay

model = deeplabv3_plus(num_classes=9)

# parameters = model.named_parameters()
# print(parameters)
# param_list = []

# for name, param in parameters:
#     print(name)
#     param_list.append(name)

# print(len(param_list))


# parameters = add_weight_decay(model)
# print(len(parameters[1]['params']))

X = torch.randn(16, 4, 512, 512)

logits = model(X)

print(logits.shape)

softmax_lab = F.softmax(logits, dim=1)

max_probs, targets_u = torch.max(softmax_lab, dim=1)
mask = max_probs.ge(.1).float()

print(max_probs)
print(targets_u)
print(mask)


#     # Compute unlabeled loss as cross entropy between strongly augmented (unlabeled) samples and previously computed
#     # pseudo-labels.
unlabeled_loss = (F.cross_entropy(logits, targets_u, reduction="none") * mask).mean()

print(unlabeled_loss)
