import logging
import numpy as np
from sklearn.metrics import confusion_matrix

logger = logging.getLogger()


class AverageMeter:
    """
    AverageMeter implements a class which can be used to track a metric over the entire training process.
    (see https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets all class variables to default values
        """
        self.val = 0
        self.vals = []
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates class variables with new value and weight
        """
        self.val = val
        self.vals.append(val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __format__(self, format):
        """
        Implements format method for printing of current AverageMeter state
        """
        return "{self.val:{format}} ({self.avg:{format}})".format(
            self=self, format=format
        )


class AverageMeterSet:
    """
    AverageMeterSet implements a class which can be used to track a set of metrics over the entire training process
    based on AverageMeters (Source: https://github.com/CuriousAI/mean-teacher/)
    """
    def __init__(self):
        self.meters = {}

    def __getitem__(self, key):
        return self.meters[key]

    def update(self, name, value, n=1):
        if name not in self.meters:
            self.meters[name] = AverageMeter()
        self.meters[name].update(value, n)

    def reset(self):
        for meter in self.meters.values():
            meter.reset()

    def values(self, postfix=""):
        return {name + postfix: meter.val for name, meter in self.meters.items()}

    def averages(self, postfix="/avg"):
        return {name + postfix: meter.avg for name, meter in self.meters.items()}

    def sums(self, postfix="/sum"):
        return {name + postfix: meter.sum for name, meter in self.meters.items()}

    def counts(self, postfix="/count"):
        return {name + postfix: meter.count for name, meter in self.meters.items()}
    
def evaluate(predictions, gts, num_classes):
    conmatrix = np.zeros((num_classes, num_classes))
    labels = np.arange(num_classes).tolist()

    #print(len(predictions))
    # dumb confusion matrix cause out of memory
    for lp, lt in zip(predictions, gts):
        #print(lp.shape)
        lp[lt == 255] = 255
        # lt[lt < 0] = -1
        conmatrix += confusion_matrix(lt.flatten(), lp.flatten(), labels=labels)


    M, N = conmatrix.shape
    tp = np.zeros(M, dtype=np.uint)
    fp = np.zeros(M, dtype=np.uint)
    fn = np.zeros(M, dtype=np.uint)

    for i in range(M):
        tp[i] = conmatrix[i, i]
        fp[i] = np.sum(conmatrix[:, i]) - tp[i]
        fn[i] = np.sum(conmatrix[i, :]) - tp[i]

    precision = tp / (tp + fp)  # = tp/col_sum
    recall = tp / (tp + fn)
    f1_score = 2 * recall * precision / (recall + precision)

    ax_p = 0  # column of confusion matrix
    # ax_t = 1  # row of confusion matrix
    acc = np.diag(conmatrix).sum() / conmatrix.sum()
    acc_cls = np.diag(conmatrix) / conmatrix.sum(axis=ax_p)
    #print(acc_cls)
    acc_cls = np.nanmean(acc_cls)
    iu = tp / (tp + fp + fn)
    mean_iu = np.nanmean(iu)
    freq = conmatrix.sum(axis=ax_p) / conmatrix.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc, np.nanmean(f1_score), iu