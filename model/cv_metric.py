from functools import reduce

import numpy as np
import pandas as pd
import torch

from logger import Loggers

Logger = Loggers.get_logger(__name__)

"""
metric functions for segment pictures
should promise that the output and target are both 0/1
"""


def normalize(output, target, th=0.25):
    if output.max() > 1:
        output = output / 255
    if target.max() > 1:
        target = target / 255
    if torch.unique(output) not in ([0], [1], [0, 1]):
        output = (output >= th)
    if torch.unique(target) not in ([0], [1], [0, 1]):
        target = (target >= th)
    return output, target


def accuracy(output, target, smooth=1e-6):
    output, target = normalize(output, target)
    y_count = reduce(lambda x, y: x * y, output.shape)
    # output_reverse = ~output
    ctp = output[target == 1].sum()
    # ctn = output[target == 0].sum()
    ctn = (~output[target == 0]).sum()
    acc = (ctp + ctn) / (y_count + smooth)
    return acc


def precision(output, target, smooth=1e-6):
    output, target = normalize(output, target)
    ctp = output[target == 1].sum()
    cfp = output[target == 0].sum()

    return ctp / (ctp + cfp + smooth)


def recall(output, target, smooth=1e-6):
    output, target = normalize(output, target)
    y_true_count = target.sum()
    ctp = output[target == 1].sum()

    return ctp / (y_true_count + smooth)


def roc_auc(output, target, smooth=1e-6):
    """
    return: TPR / FPR
    higher the better
    at least 1
    """
    output, target = normalize(output, target)
    y_true_count = target.sum()
    # noinspection PyUnresolvedReferences
    y_false_count = (~target).sum()
    ctp = output[target == 1].sum()
    cfp = output[target == 0].sum()
    # c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)

    TPR = c_recall
    FPR = cfp / (y_false_count + smooth)

    return TPR / FPR


def fbeta(output, target, th=0.5, beta=0.5, smooth=1e-6):
    """
    f_beta with numpy
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """

    output, target = normalize(output, target, th=th)
    y_true_count = target.sum()
    ctp = output[target == 1].sum()
    cfp = output[target == 0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice


class CVFbeta:
    def __init__(self, beta=0.5, smooth=1e-6):
        self.df = pd.DataFrame(columns=['threshold', 'fbeta'])
        self.best_fbeta = 0
        self.best_threshold = 0
        self.beta = beta
        self.smooth = smooth

    # noinspection PyMethodMayBeStatic
    def _fbeta_numpy(self, targets, preds):
        """
        https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
        """
        return fbeta(targets, preds, beta=self.beta, smooth=self.smooth)

    def _calc_fbeta(self, label_gt, label_pred):
        assert label_gt.shape == label_pred.shape, f'mask.shape:{label_gt.shape} != mask_pred.shape:{label_pred.shape}'
        label_gt = label_gt.astype(int).flatten()
        label_pred = label_pred.flatten()

        best_th = 0
        best_dice = 0
        # for th in np.array(range(10, 50 + 1, 5)) / 100:
        for th in np.linspace(10, 91, 9) / 100:
            # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
            # print(label_pred.max())
            dice = self._fbeta_numpy(label_gt, (label_pred >= th).astype(int), beta=0.5)
            # Logger.info(f'th: {th}, fbeta: {dice}')

            if dice > best_dice:
                best_dice = dice
                best_th = th

        ctp = label_pred[label_gt == 1].sum()
        cfp = label_pred[label_gt == 0].sum()
        ctn = (1 - label_pred)[label_gt == 0].sum()
        cfn = (1 - label_pred)[label_gt == 1].sum()
        # logger as matrix
        # Confused matrix
        recall = ctp / (ctp + cfn)
        precision = ctp / (ctp + cfp)
        accuracy = (ctp + ctn) / (ctp + ctn + cfp + cfn)

        con_mx = pd.DataFrame([[ctp, cfp], [cfn, ctn]], columns=['P', 'N'], index=['P', 'N'])

        rates = pd.DataFrame([recall, precision, accuracy], index=['recall', 'precision', 'accuracy'],
                             columns=["value"])
        Logger.info(f'Confusion matrix: \n'
                    f'{con_mx} \n'
                    f'Rates: \n'
                    f'{rates}')

        Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
        return best_dice, best_th

    # def _calc_cv(self, mask_gt, mask_pred):
    #     best_dice, best_th = self._calc_fbeta(mask_gt, mask_pred)
    #     return best_dice, best_th

    def __call__(self, mask_gt, mask_pred):
        f_beta_score, th = self._calc_fbeta(mask_gt, mask_pred)
        return f_beta_score
