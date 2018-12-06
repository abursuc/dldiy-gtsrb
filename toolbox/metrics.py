
'''
Code snippets for keeping track of evaluation metrics
'''

import numpy as np
import json


'''
                                .
                              .o8
ooo. .oo.  .oo.    .ooooo.  .o888oo  .ooooo.  oooo d8b  .oooo.o
`888P"Y88bP"Y88b  d88' `88b   888   d88' `88b `888""8P d88(  "8
 888   888   888  888ooo888   888   888ooo888  888     `"Y88b.
 888   888   888  888    .o   888 . 888    .o  888     o.  )88b
o888o o888o o888o `Y8bod8P'   "888" `Y8bod8P' d888b    8""888P'
'''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def value(self):
        return self.avg

class SumMeter(object):
    """Computes and stores the sum and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    def value(self):
        return self.sum


class ValueMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0

    def update(self, val):
        self.val = val

    def value(self):
        return self.val



class ConfusionMeter(object):
    """
    The ConfusionMeter constructs a confusion matrix for a multi-class
    classification problems. It does not support multi-label, multi-class problems:
    for such problems, please use MultiLabelConfusionMeter.
    """

    def __init__(self, k, normalized=False):
        """
        Args:
            k (int): number of classes in the classification problem
            normalized (boolean): Determines whether or not the confusion matrix
                is normalized or not
        """
        self.conf = np.ndarray((k, k), dtype=np.int32)
        self.normalized = normalized
        self.k = k
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def update(self, predicted, target):
        """
        Computes the confusion matrix of K x K size where K is no of classes
        Args:
            predicted (tensor): Can be an N x K tensor of predicted scores obtained from
                the model for N examples and K classes or an N-tensor of
                integer values between 0 and K-1.
            target (tensor): Can be a N-tensor of integer values assumed to be integer
                values between 0 and K-1 or N x K tensor, where targets are
                assumed to be provided as one-hot vectors
        """
        if predicted.numel()>1:
            predicted.squeeze_()
            target.squeeze_()
        else:
            predicted = predicted.view(-1)
            target = target.view(-1)

        predicted = predicted.to('cpu').numpy()
        target = target.to('cpu').numpy()

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.k, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 1 and k'

        onehot_target = np.ndim(target) != 1
        if onehot_target:
            assert target.shape[1] == self.k, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (predicted.max() < self.k) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.k * target
        bincount_2d = np.bincount(x.astype(np.int32),
                                  minlength=self.k ** 2)
        assert bincount_2d.size == self.k ** 2
        conf = bincount_2d.reshape((self.k, self.k))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            res = conf / conf.sum(1).clip(min=1e-12)[:, None]
            return res.tolist()
        else:
            return self.conf.tolist()


def make_meters(num_classes=2):
    meters_dict = {
        'loss': AverageMeter(),
        'acc1': AverageMeter(),
        'mAP': AverageMeter(),
        'meanIoU': ValueMeter(),
        'acc_class': ValueMeter(),
        'fwavacc': ValueMeter(),
        'batch_time': AverageMeter(),
        'data_time': AverageMeter(),
        'epoch_time': SumMeter(),
        'confusion_matrix': ConfusionMeter(num_classes),
    }
    return meters_dict



def save_meters(meters, fn, epoch=0):

    logged = {}
    for name, meter in meters.items():
        logged[name] = meter.value()

    if epoch > 0:
        logged['epoch'] = epoch

    print(f'Saving meters to {fn}')
    with open(fn, 'w') as f:
        json.dump(logged, f)


'''
 .oooo.o  .ooooo.   .ooooo.  oooo d8b  .ooooo.   .oooo.o
d88(  "8 d88' `"Y8 d88' `88b `888""8P d88' `88b d88(  "8
`"Y88b.  888       888   888  888     888ooo888 `"Y88b.
o.  )88b 888   .o8 888   888  888     888    .o o.  )88b
8""888P' `Y8bod8P' `Y8bod8P' d888b    `Y8bod8P' 8""888P'
'''

def evaluate(hist):
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)+1e-10)
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / (hist.sum()+ 1e-10)
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def accuracy_classif(output, target, topk=1):
    """Computes the precision@k for the specified values of k"""
    maxk = 1
    # maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # res = []
    correct_k = correct[:maxk].view(-1).float().sum(0)
    correct_k.mul_(1.0 / batch_size)
    res = correct_k.clone()

    return res.item(), pred, target


def accuracy_regression(output, target):
    mae_mean = (output - target).abs().mean()
    mae_std = (output - target).abs().std()
    mse_mean = (output - target).pow(2).mean()
    mse_std = (output - target).pow(2).std()
    rmse = (output - target).pow(2).mean().sqrt()
    return mae, mse, rmse
    return mae_mean.item(), mae_std.item(), mse_mean.item(), mse_std.item(), rmse_mean.item(), rmse_std.item()


def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(
        n * label[k].astype(int) + pred[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist)+ 1e-10)


