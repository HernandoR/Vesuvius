import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F


def normalize(output, target, th=0.25):
    if output.max() > 1:
        output = output / 255
    if target.max() > 1:
        target = target / 255
    if torch.unique(output) not in ([0.], [1.], [0., 1.]):
        output = (output >= th)
    if torch.unique(target) not in ([0.], [1.], [0., 1.]):
        target = (target >= th)
    return output, target


def nll_loss(output, target):
    return F.nll_loss(output, target)


def BCELoss(y_pred, y_true):
    y_pred,y_true=y_pred.squeeze(),y_true.squeeze()
    assert y_pred.shape == y_true.shape, f'y_pred.shape:{y_pred.shape} != y_true.shape:{y_true.shape}'
    # return nn.BCEWithLogitsLoss()(y_pred, y_true)

    return smp.losses.SoftBCEWithLogitsLoss()(y_pred, y_true)


def MCCLoss(y_pred, y_true):
    # y_pred,y_true=(y_pred, y_true)
    assert y_pred.shape == y_true.shape, f'y_pred.shape:{y_pred.shape} != y_true.shape:{y_true.shape}'
    return smp.losses.MCCLoss()(y_pred, y_true)
