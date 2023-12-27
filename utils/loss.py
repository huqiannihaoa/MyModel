# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Loss functions
"""

import torch
import torch.nn as nn

from utils.metrics import bbox_iou
from utils.torch_utils import de_parallel

from utils.metrics import bbox_iou_xiou
from utils.general import bbox_iou_eciou

def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super().__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)  # prob from logits
        dx = pred - true  # reduce only missing label effects
        # dx = (pred - true).abs()  # reduce missing label and false label effects
        alpha_factor = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss
# class ComputeLoss:
#     # Compute losses
#     def __init__(self, model, loss_category, autobalance=False):
#         self.sort_obj_iou = False
#         self.loss_category = loss_category
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters
#         print('您现在使用的损失函数是：' + loss_category)
#
#         # Define criteria
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
#
#         # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
#
#         # Focal loss
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
#
#         det = de_parallel(model).model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         for k in 'na', 'nc', 'nl', 'anchors':
#             setattr(self, k, getattr(det, k))
#
#     def __call__(self, p, targets):  # predictions, targets, model
#         device = targets.device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
#
#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#
#             n = b.shape[0]  # number of targets
#             if n:
#                 ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
#
#                 # Regression
#                 pxy = ps[:, :2].sigmoid() * 2 - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 if self.loss_category == 'CIoU':
#                     iou = bbox_iou(pbox, tbox[i], CIoU=True, Focal=False, scale=False)
#                 elif self.loss_category == 'SIoU':
#                     iou = bbox_iou(pbox, tbox[i], SIoU=True, Focal=False, scale=False)
#                 elif self.loss_category == 'DIoU':
#                     iou = bbox_iou(pbox, tbox[i], DIoU=True, Focal=False, scale=False)
#                 elif self.loss_category == 'EIoU':
#                     iou = bbox_iou(pbox, tbox[i], EIoU=True, Focal=False, scale=False)
#                 elif self.loss_category == 'GIoU':
#                     iou = bbox_iou(pbox, tbox[i], GIoU=True, Focal=False, scale=False)
#                 elif self.loss_category == 'WIoU':
#                     iou = bbox_iou(pbox, tbox[i], WIoU=True, Focal=False, scale=True)
#                 elif self.loss_category == 'Focal_CIoU':
#                     iou = bbox_iou(pbox, tbox[i], CIoU=True, Focal=True, scale=False)
#                 elif self.loss_category == 'Focal_DIoU':
#                     iou = bbox_iou(pbox, tbox[i], DIoU=True, Focal=True, scale=False)
#                 elif self.loss_category == 'Focal_EIoU':
#                     iou = bbox_iou(pbox, tbox[i], EIoU=True, Focal=True, scale=False)
#                 elif self.loss_category == 'Focal_GIoU':
#                     iou = bbox_iou(pbox, tbox[i], GIoU=True, Focal=True, scale=False)
#                 elif self.loss_category == 'Focal_SIoU':
#                     iou = bbox_iou(pbox, tbox[i], SIoU=True, Focal=True, scale=False)
#
#                 if type(iou) is tuple:
#                     if len(iou) == 2:
#                         lbox += (iou[1].detach().squeeze() * (1 - iou[0].squeeze())).mean()
#                         iou = iou[0].squeeze()
#                     else:
#                         lbox += (iou[0] * iou[1]).mean()
#                         iou = iou[2].squeeze()
#                 else:
#                     lbox += (1.0 - iou.squeeze()).mean()
#                     iou = iou.squeeze()
#
#                 # Objectness
#                 score_iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     sort_id = torch.argsort(score_iou)
#                     b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
#                 tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
#
#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += self.BCEcls(ps[:, 5:], t)  # BCE
#
#                 # Append targets to text file
#                 # with open('targets.txt', 'a') as file:
#                 #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]
#
#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
#
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj']
#         lcls *= self.hyp['cls']
#         bs = tobj.shape[0]  # batch size
#
#         return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()
#
#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
#
#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
#                             ], device=targets.device).float() * g  # offsets
#
#         for i in range(self.nl):
#             anchors = self.anchors[i]
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#
#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter
#
#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1 < g) & (gxy > 1)).T
#                 l, m = ((gxi % 1 < g) & (gxi > 1)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0
#
#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices
#
#             # Append
#             a = t[:, 6].long()  # anchor indices
#             indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))  # image, anchor, grid indices
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class
#
#         return tcls, tbox, indices, anch
import torch
from torch import nn

class VFLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(VFLoss, self).__init__()
        # 传递 nn.BCEWithLogitsLoss() 损失函数  must be nn.BCEWithLogitsLoss()
        self.loss_fcn = loss_fcn  #
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply VFL to each element

    def forward(self, pred, true):

        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits

        focal_weight = true * (true > 0.0).float() + self.alpha * (pred_prob - true).abs().pow(self.gamma) * (
                    true <= 0.0).float()
        loss *= focal_weight

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        pred_prob = torch.sigmoid(pred)  # prob from logits
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = torch.abs(true - pred_prob) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class ComputeLoss:
    sort_obj_iou = False

    # Compute losses
    def __init__(self, model, autobalance=False):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets

        # Focal loss
        g = h['fl_gamma']  # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
            # BCEcls, BCEobj = VFLoss(BCEcls, g), VFLoss(BCEobj, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.ssi = list(m.stride).index(16) if autobalance else 0  # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
        self.na = m.na  # number of anchors
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.anchors = m.anchors
        self.device = device

    def __call__(self, p, targets):  # predictions, targets
        lcls = torch.zeros(1, device=self.device)  # class loss
        lbox = torch.zeros(1, device=self.device)  # box loss
        lobj = torch.zeros(1, device=self.device)  # object loss
        tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = torch.zeros(pi.shape[:4], dtype=pi.dtype, device=self.device)  # target obj

            n = b.shape[0]  # number of targets
            if n:
                # pxy, pwh, _, pcls = pi[b, a, gj, gi].tensor_split((2, 4, 5), dim=1)  # faster, requires torch 1.8.0
                pxy, pwh, _, pcls = pi[b, a, gj, gi].split((2, 2, 1, self.nc), 1)  # target-subset of predictions

                # Regression
                pxy = pxy.sigmoid() * 2 - 0.5
                pwh = (pwh.sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1)  # predicted box
                #iou = bbox_iou(pbox, tbox[i], CIoU=True).squeeze()  # iou(prediction, target)


                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False, EIoU=True)
                # iou = bbox_iou(pbox, tbox[i], EIoU=True, Focal=False)  # iou(prediction, target)
                # iou = bbox_iou(pbox, tbox[i], CIoU=True, Focal=False)  # iou(prediction, target)


                # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, EIoU=True)
                iou = bbox_iou_eciou(pbox.T, tbox[i], x1y1x2y2=False, EfficiCIoULoss=True)


                # iou = bbox_iou(pbox, tbox[i], EIoU=True, Focal=False)  # iou(prediction, target)
                # iou = bbox_iou_xiou(pbox.T, tbox[i], x1y1x2y2=False, XIoU=True)

                # iou(prediction, target)

                # lbox += (1.0 - iou).mean()  # iou loss
                if type(iou) is tuple:  # 如果 iou 是一个元组类型
                    if len(iou) == 2:  # 如果元组的长度为2
                        # 对变量 lbox 进行加法运算，加上 iou[1] 的平均值乘以 (1 - iou[0].squeeze())
                        lbox += (iou[1].detach().squeeze() * (1 - iou[0].squeeze())).mean()
                        # 更新变量 iou 为 iou[0].squeeze()
                        iou = iou[0].squeeze()
                    else:  # 如果元组的长度不为2
                        # 对变量 lbox 进行加法运算，加上 iou[0] 乘以 iou[1] 的平均值
                        lbox += (iou[0] * iou[1]).mean()
                        # 更新变量 iou 为 iou[2].squeeze()
                        iou = iou[2].squeeze()
                else:  # 如果 iou 不是元组类型
                    # 对变量 lbox 进行加法运算，加上 (1.0 - iou.squeeze()) 的平均值
                    lbox += (1.0 - iou.squeeze()).mean()
                    # 更新变量 iou 为 iou.squeeze()
                    iou = iou.squeeze()

                # Objectness
                iou = iou.detach().clamp(0).type(tobj.dtype)
                if self.sort_obj_iou:
                    j = iou.argsort()
                    b, a, gj, gi, iou = b[j], a[j], gj[j], gi[j], iou[j]
                if self.gr < 1:
                    iou = (1.0 - self.gr) + self.gr * iou
                tobj[b, a, gj, gi] = iou  # iou ratio

                # Classification
                if self.nc > 1:  # cls loss (only if multiple classes)
                    t = torch.full_like(pcls, self.cn, device=self.device)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += self.BCEcls(pcls, t)  # BCE

                # Append targets to text file
                # with open('targets.txt', 'a') as file:
                #     [file.write('%11.5g ' * 4 % tuple(x) + '\n') for x in torch.cat((txy[i], twh[i]), 1)]

            obji = self.BCEobj(pi[..., 4], tobj)
            lobj += obji * self.balance[i]  # obj loss
            if self.autobalance:
                self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()

        if self.autobalance:
            self.balance = [x / self.balance[self.ssi] for x in self.balance]
        lbox *= self.hyp['box']
        lobj *= self.hyp['obj']
        lcls *= self.hyp['cls']
        bs = tobj.shape[0]  # batch size

        return (lbox + lobj + lcls) * bs, torch.cat((lbox, lobj, lcls)).detach()

    def build_targets(self, p, targets):
        # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
        na, nt = self.na, targets.shape[0]  # number of anchors, targets
        tcls, tbox, indices, anch = [], [], [], []
        gain = torch.ones(7, device=self.device)  # normalized to gridspace gain
        ai = torch.arange(na, device=self.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
        targets = torch.cat((targets.repeat(na, 1, 1), ai[..., None]), 2)  # append anchor indices

        g = 0.5  # bias
        off = torch.tensor(
            [
                [0, 0],
                [1, 0],
                [0, 1],
                [-1, 0],
                [0, -1],  # j,k,l,m
                # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
            ],
            device=self.device).float() * g  # offsets

        for i in range(self.nl):
            anchors, shape = self.anchors[i], p[i].shape
            gain[2:6] = torch.tensor(shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain  # shape(3,n,7)
            if nt:
                # Matches
                r = t[..., 4:6] / anchors[:, None]  # wh ratio
                j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
                # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
                t = t[j]  # filter

                # Offsets
                gxy = t[:, 2:4]  # grid xy
                gxi = gain[[2, 3]] - gxy  # inverse
                j, k = ((gxy % 1 < g) & (gxy > 1)).T
                l, m = ((gxi % 1 < g) & (gxi > 1)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            bc, gxy, gwh, a = t.chunk(4, 1)  # (image, class), grid xy, grid wh, anchors
            a, (b, c) = a.long().view(-1), bc.long().T  # anchors, image, class
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid indices

            # Append
            indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
            tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
            anch.append(anchors[a])  # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch




from utils.general import box_iou_repulsion
import numpy as np


def IoG_(gt_box, pre_box):
    inter_xmin = torch.max(gt_box[:, 0], pre_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pre_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pre_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pre_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G


def smooth_ln(x, deta=0.5):
    return torch.where(
        torch.le(x, deta),
        -torch.log(1 - x),
        ((x - deta) / (1 - deta)) - np.log(1 - deta)
    )


def repulsion(pbox, gtbox, deta=0.5, pnms=0.1, gtnms=0.1, x1x2y1y2=False):
    repgt_loss = 0.0
    repbox_loss = 0.0
    pbox = pbox.detach()
    gtbox = gtbox.detach()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gtbox_cpu = gtbox.cuda().data.cpu().numpy()
    pgiou = box_iou_repulsion(pbox, gtbox, x1y1x2y2=x1x2y1y2)
    pgiou = pgiou.cuda().data.cpu().numpy()
    ppiou = box_iou_repulsion(pbox, pbox, x1y1x2y2=x1x2y1y2)
    ppiou = ppiou.cuda().data.cpu().numpy()
    len = pgiou.shape[0]
    for j in range(len):
        for z in range(j, len):
            ppiou[j, z] = 0
            if (gtbox_cpu[j][0] == gtbox_cpu[z][0]) and (gtbox_cpu[j][1] == gtbox_cpu[z][1]) and (
                    gtbox_cpu[j][2] == gtbox_cpu[z][2]) and (gtbox_cpu[j][3] == gtbox_cpu[z][3]):
                pgiou[j, z] = 0
                pgiou[z, j] = 0
                ppiou[z, j] = 0
    pgiou = torch.from_numpy(pgiou).cuda().detach()
    ppiou = torch.from_numpy(ppiou).cuda().detach()
    max_iou, argmax_iou = torch.max(pgiou, 1)
    pg_mask = torch.gt(max_iou, gtnms)
    num_repgt = pg_mask.sum()
    if num_repgt > 0:
        iou_pos = pgiou[pg_mask, :]
        max_iou_sec, argmax_iou_sec = torch.max(iou_pos, 1)
        pbox_sec = pbox[pg_mask, :]
        gtbox_sec = gtbox[argmax_iou_sec, :]
        IOG = IoG_(gtbox_sec, pbox_sec)
        repgt_loss = smooth_ln(IOG, deta)
        repgt_loss = repgt_loss.mean()
    pp_mask = torch.gt(ppiou, pnms)
    num_pbox = pp_mask.sum()
    if num_pbox > 0:
        repbox_loss = smooth_ln(ppiou, deta)
        repbox_loss = repbox_loss.mean()
    torch.cuda.empty_cache()

    return repgt_loss, repbox_loss


# class ComputeLoss:
#     def __init__(self, model, autobalance=False):
#         self.sort_obj_iou = False
#         device = next(model.parameters()).device  # get model device
#         h = model.hyp  # hyperparameters
#
#         BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
#         BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))
#
#         self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))  # positive, negative BCE targets
#         g = h['fl_gamma']  # focal loss gamma
#         if g > 0:
#             BCEcls, BCEobj = FocalLoss(BCEcls, g), FocalLoss(BCEobj, g)
#
#         det = de_parallel(model).model[-1]  # Detect() module
#         self.balance = {3: [4.0, 1.0, 0.4]}.get(det.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
#         self.ssi = list(det.stride).index(16) if autobalance else 0  # stride 16 index
#         self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, 1.0, h, autobalance
#         for k in 'na', 'nc', 'nl', 'anchors':
#             setattr(self, k, getattr(det, k))
#
#     def __call__(self, p, targets):  # predictions, targets, model
#         device = targets.device
#         lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
#         lrepBox, lrepGT = torch.zeros(1, device=device), torch.zeros(1, device=device)  # update
#         tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets
#         # Losses
#         for i, pi in enumerate(p):  # layer index, layer predictions
#             b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
#             tobj = torch.zeros_like(pi[..., 0], device=device)  # target obj
#
#             n = b.shape[0]  # number of targets
#             if n:
#                 ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets
#                 # Regression
#                 pxy = ps[:, :2].sigmoid() * 2 - 0.5
#                 pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
#                 pbox = torch.cat((pxy, pwh), 1)  # predicted box
#                 # iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, EIoU=True)
#                 iou = bbox_iou(pbox, tbox[i],SIoU=True, Focal=False)  # iou(prediction, target)
#                 # iou = bbox_iou_xiou(pbox.T, tbox[i], x1y1x2y2=False, EIoU=True)
#                 lbox += (1.0 - iou).mean()
#                 # Objectness
#                 score_iou = iou.detach().clamp(0).type(tobj.dtype)
#                 if self.sort_obj_iou:
#                     sort_id = torch.argsort(score_iou)
#                     b, a, gj, gi, score_iou = b[sort_id], a[sort_id], gj[sort_id], gi[sort_id], score_iou[sort_id]
#                 tobj[b, a, gj, gi] = (1.0 - self.gr) + self.gr * score_iou  # iou ratio
#                 # Classification
#                 if self.nc > 1:  # cls loss (only if multiple classes)
#                     t = torch.full_like(ps[:, 5:], self.cn, device=device)  # targets
#                     t[range(n), tcls[i]] = self.cp
#                     lcls += self.BCEcls(ps[:, 5:], t)  # BCE
#
#                 dic = {index: [] for index in range(256)}
#                 for indexs, value in enumerate(b):
#                     dic[int(value)].append(indexs)
#                 bts = 0  # update
#                 deta = 0.5  # smooth_ln parameter
#                 Rp_nms = 0.1  # RepGT loss nms
#                 _lrepGT = 0.0
#                 _lrepBox = 0.0
#                 for id, indexs in dic.items():
#                     if indexs:
#                         lrepgt, lrepbox = repulsion(pbox[indexs], tbox[i][indexs], deta=deta, pnms=Rp_nms, gtnms=Rp_nms)
#                         _lrepGT += lrepgt
#                         _lrepBox += lrepbox
#                         bts += 1
#                 if bts > 0:
#                     _lrepGT /= bts
#                     _lrepBox /= bts
#                 lrepGT += _lrepGT
#                 lrepBox += _lrepBox
#             # update
#             obji = self.BCEobj(pi[..., 4], tobj)
#             lobj += obji * self.balance[i]  # obj loss
#             if self.autobalance:
#                 self.balance[i] = self.balance[i] * 0.9999 + 0.0001 / obji.detach().item()
#         if self.autobalance:
#             self.balance = [x / self.balance[self.ssi] for x in self.balance]
#         lbox *= self.hyp['box']
#         lobj *= self.hyp['obj']
#         lcls *= self.hyp['cls']
#         lrep = 0.01 * lrepGT / 3.0 + 0.1 * lrepBox / 3.0
#         bs = tobj.shape[0]  # batch size, gpu
#
#         # no lrep, loss
#         loss = lbox + lobj + lcls + lrep
#         return loss * bs, torch.cat((lbox, lobj, lcls)).detach()
#
#     def build_targets(self, p, targets):
#         # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
#         na, nt = self.na, targets.shape[0]  # number of anchors, targets
#         tcls, tbox, indices, anch = [], [], [], []
#         gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
#         ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt)  # same as .repeat_interleave(nt)
#         targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2)  # append anchor indices
#
#         g = 0.5  # bias
#         off = torch.tensor([[0, 0],
#                             [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
#                             ], device=targets.device).float() * g  # offsets
#
#         for i in range(self.nl):
#             anchors, shape = self.anchors[i], p[i].shape
#             gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain
#
#             # Match targets to anchors
#             t = targets * gain
#             if nt:
#                 # Matches
#                 r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
#                 j = torch.max(r, 1 / r).max(2)[0] < self.hyp['anchor_t']  # compare
#                 # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
#                 t = t[j]  # filter
#
#                 # Offsets
#                 gxy = t[:, 2:4]  # grid xy
#                 gxi = gain[[2, 3]] - gxy  # inverse
#                 j, k = ((gxy % 1 < g) & (gxy > 1)).T
#                 l, m = ((gxi % 1 < g) & (gxi > 1)).T
#                 j = torch.stack((torch.ones_like(j), j, k, l, m))
#                 t = t.repeat((5, 1, 1))[j]
#                 offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
#             else:
#                 t = targets[0]
#                 offsets = 0
#
#             # Define
#             b, c = t[:, :2].long().T  # image, class
#             gxy = t[:, 2:4]  # grid xy
#             gwh = t[:, 4:6]  # grid wh
#             gij = (gxy - offsets).long()
#             gi, gj = gij.T  # grid xy indices
#
#             # Append
#             a = t[:, 6].long()  # anchor indices
#             indices.append((b, a, gj.clamp_(0, shape[2] - 1), gi.clamp_(0, shape[3] - 1)))  # image, anchor, grid
#             tbox.append(torch.cat((gxy - gij, gwh), 1))  # box
#             anch.append(anchors[a])  # anchors
#             tcls.append(c)  # class
#
#         return tcls, tbox, indices, anch

