import torch
from torchmetrics import JaccardIndex

# # custom mIoU class
# from torch import Tensor
# from torchmetrics import Metric
# class mIoU(Metric):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.num_classes = 9
#         self.add_state("intersection", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")
#         self.add_state("union", default=torch.zeros(self.num_classes), dist_reduce_fx="sum")

#     def update(self, preds: Tensor, target: Tensor) -> None:

#         # for each class, calculate intersection and union
#         for i in range(self.num_classes):
#             preds_i = preds == i
#             target_i = target == i
#             self.intersection[i] += torch.sum(torch.logical_and(preds_i, target_i))
#             self.union[i] += torch.sum(torch.logical_or(preds_i, target_i))

#     def compute(self) -> tuple:
#         # calculate IoU for each class
#         ious = torch.Tensor([(self.intersection[i] / self.union[i]).float() if self.union[i] != 0 else 0 for i in range(self.num_classes)])

#         # calculate mIoU
#         miou = torch.sum(ious).float() / self.num_classes
        
#         return (miou, ious)
    
class Metrics():
    def __init__(self, args):
        self.num_classes = 9
        self.labeled_mIoU = JaccardIndex(task='multiclass', num_classes=self.num_classes).to(args.device)
        self.labeled_IoU = [JaccardIndex(task="binary").to(args.device) for i in range(self.num_classes)]
        self.unlabeled_mIoU = JaccardIndex(task='multiclass', num_classes=self.num_classes).to(args.device)
        self.unlabeled_IoU = [JaccardIndex(task="binary").to(args.device) for i in range(self.num_classes)]
        self.val_mIoU = JaccardIndex(task='multiclass', num_classes=self.num_classes)
        self.val_IoU = [JaccardIndex(task="binary") for i in range(self.num_classes)]

    def __repr__(self):
        return self.print()
    def __str__(self):
        return self.print()

    def update_labeled(self, logits, labels):
        pred = torch.argmax(logits, dim=1)
        self.labeled_mIoU.update(pred, labels)
        for i in range(self.num_classes):
            self.labeled_IoU[i].update(pred==i, labels==i)

    def update_unlabeled(self, logits, labels):
        pred = torch.argmax(logits, dim=1)
        self.unlabeled_mIoU.update(pred, labels)
        for i in range(self.num_classes):
            self.unlabeled_IoU[i].update(pred==i, labels==i)

    def update_validation(self, logits, labels):
        pred = torch.argmax(logits, dim=1)
        self.val_mIoU.update(pred, labels)
        for i in range(self.num_classes):
            self.val_IoU[i].update(pred==i, labels==i)

    def compute(self):
        labeled_mIoU = self.labeled_mIoU.compute()
        labeled_IoU = [i.compute() for i in self.labeled_IoU]
        unlabeled_mIoU = self.unlabeled_mIoU.compute()
        unlabeled_IoU = [i.compute() for i in self.unlabeled_IoU]
        val_mIoU = self.val_mIoU.compute()
        val_IoU = [i.compute() for i in self.val_IoU]
        return labeled_mIoU, labeled_IoU, unlabeled_mIoU, unlabeled_IoU, val_mIoU, val_IoU

    def reset(self):
        self.labeled_mIoU.reset()
        [i.reset() for i in self.labeled_IoU]
        self.unlabeled_mIoU.reset()
        [i.reset() for i in self.unlabeled_IoU]
        self.val_mIoU.reset()
        [i.reset() for i in self.val_IoU]

    def print(self):
        labeled_mIoU, labeled_IoU, unlabeled_mIoU, unlabeled_IoU, val_mIoU, val_IoU = self.compute()
    
        labeled_metrics_str = "Labeled IoU:\n"
        labeled_metrics_str += f"mIoU: {labeled_mIoU:.4f}\n"
        for i, iou in enumerate(labeled_IoU):
            if i % 5 == 4:
                labeled_metrics_str += f"IoU{i}: {iou:.4f}\n"
            else:
                labeled_metrics_str += f"IoU{i}: {iou:.4f}\t"
    
        unlabeled_metrics_str = "Unlabeled IoU:\n"
        unlabeled_metrics_str += f"mIoU: {unlabeled_mIoU:.4f}\n"
        for i, iou in enumerate(unlabeled_IoU):
            if i % 5 == 4:
                unlabeled_metrics_str += f"IoU{i}: {iou:.4f}\n"
            else:
                unlabeled_metrics_str += f"IoU{i}: {iou:.4f}\t"

        val_metrics_str = "Validation IoU:\n"
        val_metrics_str += f"mIoU: {val_mIoU:.4f}\n"
        for i, iou in enumerate(val_IoU):
            if i % 5 == 4:
                val_metrics_str += f"IoU{i}: {iou:.4f}\n"
            else:
                val_metrics_str += f"IoU{i}: {iou:.4f}\t"
    
        return labeled_metrics_str + "\n" + unlabeled_metrics_str + "\n" + val_metrics_str