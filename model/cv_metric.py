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


def normalize(tens, th=0.25):
    if tens.max() > 1:
        tens = tens.sigmoid()
    # if target.max() > 1:
    #     target = target / 255
    if torch.unique(tens) not in ([0], [1], [0, 1]):
        # Logger.info(f'unique output:{torch.unique(output)}')
        tens = (tens >= th)
    # if torch.unique(target) not in ([0], [1], [0, 1]):
    #     # Logger.info(f'unique target:{torch.unique(target)}')
    #     target = (target >= th)
    # tens, target = tens.to(torch.uint8), target.to(torch.uint8)
    tens = tens.to(torch.uint8)
    return tens


def accuracy(output, target, smooth=1e-6):
    output = normalize(output)
    target = normalize(target)
    true_positives = (output & target).sum().item()
    total_samples = target.numel()
    accuracy_score = true_positives / (total_samples + smooth)
    return accuracy_score


def precision(output, target, smooth=1e-6):
    output = normalize(output)
    target = normalize(target)
    true_positives = (output & target).sum().item()
    false_positives = (output & ~target).sum().item()
    precision_score = true_positives / (true_positives + false_positives + smooth)
    return precision_score


def recall(output, target, smooth=1e-6):
    output = normalize(output)
    target = normalize(target)
    true_positives = (output & target).sum().item()
    false_negatives = (~output & target).sum().item()
    recall_score = true_positives / (true_positives + false_negatives + smooth)
    return recall_score


def roc_auc(output, target, smooth=1e-6):
    """
    return: TPR / FPR
    higher the better
    at least 1
    """
    output = normalize(output)
    target = normalize(target)
    true_positives = (output & target).sum().item()
    false_positives = (output & ~target).sum().item()
    true_negatives = (~output & ~target).sum().item()
    false_negatives = (~output & target).sum().item()
    sensitivity = true_positives / (true_positives + false_negatives + smooth)
    specificity = true_negatives / (true_negatives + false_positives + smooth)
    roc_auc_score = (sensitivity + specificity) / 2
    return roc_auc_score


def fbeta(output, target, th=0.25, beta=0.5, smooth=1e-6):
    """
    f_beta with numpy
    https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection/discussion/397288
    """

    # output, target = normalize(output, target, th=th)
    output = normalize(output, th=th)
    target = normalize(target, th=th)
    # Calculate true positives, false positives, and false negatives
    true_positives = (output * target).sum()
    false_positives = (output * (1 - target)).sum()
    false_negatives = ((1 - output) * target).sum()

    # Calculate precision and recall
    precision = true_positives / (true_positives + false_positives + smooth)
    recall = true_positives / (true_positives + false_negatives + smooth)

    # Calculate F-beta score
    f_beta = (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall + smooth)

    #
    # y_true_count = target.sum()
    # ctp = output[target == 1].sum()
    # cfp = output[target == 0].sum()
    # beta_squared = beta * beta
    #
    # c_precision = ctp / (ctp + cfp + smooth)
    # c_recall = ctp / (y_true_count + smooth)
    # dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return f_beta


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
