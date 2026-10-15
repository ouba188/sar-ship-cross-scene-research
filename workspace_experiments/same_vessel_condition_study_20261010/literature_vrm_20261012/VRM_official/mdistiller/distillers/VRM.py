# VRM
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ._base import Distiller
import matplotlib.pyplot as plt
from ._common import ConvReg, get_feat_shapes, get_pooled_feat_shapes, MLPReg

def isv_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=50):  # f_s: (B,100),  f_t: (B,100)
    stu_w = f_sw.view(f_sw.shape[0], -1); stu_s = f_ss.view(f_ss.shape[0], -1) # (B,100)
    tea_w = f_tw.view(f_tw.shape[0], -1); tea_s = f_ts.view(f_ts.shape[0], -1) # (B,100)

    ########### ZSNorm of Logits ###########
    stu_w_mean = stu_w.mean(dim=-1, keepdim=True); stu_s_mean = stu_s.mean(dim=-1, keepdim=True) # (B,1) # mean is sample-wise
    tea_w_mean = tea_w.mean(dim=-1, keepdim=True); tea_s_mean = tea_s.mean(dim=-1, keepdim=True) # (B,1)
    stu_w_std = stu_w.std(dim=-1, keepdim=True); stu_s_std = stu_s.std(dim=-1, keepdim=True) # (B,1) # std is sample-wise
    tea_w_std = tea_w.std(dim=-1, keepdim=True); tea_s_std = tea_s.std(dim=-1, keepdim=True) # (B,1)
    stu_w = (stu_w - stu_w_mean) / (1e-7 + stu_w_std); stu_s = (stu_s - stu_s_mean) / (1e-7 + stu_s_std)
    tea_w = (tea_w - tea_w_mean) / (1e-7 + tea_w_std); tea_s = (tea_s - tea_s_mean) / (1e-7 + tea_s_std)

    ########### Convert Logits to Probs ###########
    T = 4.0
    stu_w = F.softmax(stu_w / T, dim=1); stu_s = F.softmax(stu_s / T, dim=1)
    tea_w = F.softmax(tea_w / T, dim=1); tea_s = F.softmax(tea_s / T, dim=1)

    ########## RKD Angle loss ##########
    with torch.no_grad():
        # compute pairwise L1 distance within batch
        td = tea_w.unsqueeze(0) - tea_s.unsqueeze(1) # (B,B,256)
        norm_td = F.normalize(td, p=2, dim=2) # (B,B,256) - normalise dim 2 using L2 distance
    # compute pairwise L1 distance within batch
    sd = stu_w.unsqueeze(0) - stu_s.unsqueeze(1) # (B,B,256)
    norm_sd = F.normalize(sd, p=2, dim=2) # (B,B,256)
    loss_isv = F.smooth_l1_loss(norm_td, norm_sd) * edge_weight # (B,B,256)
    return loss_isv
#
def icv_loss(f_sw, f_ss, f_tw, f_ts, edge_weight=50):  # f_s: (B,100),  f_t: (B,100)
    stu_w = f_sw.view(f_sw.shape[0], -1); stu_s = f_ss.view(f_ss.shape[0], -1) # (B,100)
    tea_w = f_tw.view(f_tw.shape[0], -1); tea_s = f_ts.view(f_ts.shape[0], -1) # (B,100)

    ########### ZSNorm of Logits ###########
    stu_w_mean = stu_w.mean(dim=-1, keepdim=True); stu_s_mean = stu_s.mean(dim=-1, keepdim=True) # (B,1) # mean is sample-wise
    tea_w_mean = tea_w.mean(dim=-1, keepdim=True); tea_s_mean = tea_s.mean(dim=-1, keepdim=True) # (B,1)
    stu_w_std = stu_w.std(dim=-1, keepdim=True); stu_s_std = stu_s.std(dim=-1, keepdim=True) # (B,1) # std is sample-wise
    tea_w_std = tea_w.std(dim=-1, keepdim=True); tea_s_std = tea_s.std(dim=-1, keepdim=True) # (B,1)
    stu_w = (stu_w - stu_w_mean) / (1e-7 + stu_w_std); stu_s = (stu_s - stu_s_mean) / (1e-7 + stu_s_std)
    tea_w = (tea_w - tea_w_mean) / (1e-7 + tea_w_std); tea_s = (tea_s - tea_s_mean) / (1e-7 + tea_s_std)

    ########### Convert Logits to Probs ###########
    T = 4.0
    stu_w = F.softmax(stu_w / T, dim=1); stu_s = F.softmax(stu_s / T, dim=1)
    tea_w = F.softmax(tea_w / T, dim=1); tea_s = F.softmax(tea_s / T, dim=1)

    ########## RKD Angle loss ##########
    with torch.no_grad():
        # compute pairwise L1 distance within batch
        td = (tea_w.unsqueeze(2) - tea_s.unsqueeze(1)).permute(1, 2, 0) # (B,B,256)
        norm_td = F.normalize(td, p=2, dim=2) # (B,B,256) - normalise dim 2 using L2 distance
    # compute pairwise L1 distance within batch
    sd = (stu_w.unsqueeze(2) - stu_s.unsqueeze(1)).permute(1, 2, 0) # (B,B,256)
    norm_sd = F.normalize(sd, p=2, dim=2) # (B,B,256)
    loss_a = F.smooth_l1_loss(norm_td, norm_sd)
    loss_icv = edge_weight * loss_a
    return loss_icv

class VRM(Distiller):
    def __init__(self, student, teacher, cfg):
        super(VRM, self).__init__(student, teacher)
        self.temperature = cfg.KD.TEMPERATURE # 4
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT # 0.1
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT # 0.9

    def forward_train(self, image_weak, image_strong, target, **kwargs): # image_weak & image_strong (64,3,32,32), target (64,)c
        logits_student_weak, _ = self.student(image_weak) # (64,100)
        logits_student_strong, _ = self.student(image_strong) # (64,100)
        with torch.no_grad():
            logits_teacher_weak, _ = self.teacher(image_weak) # (64,100)
            logits_teacher_strong, _ = self.teacher(image_strong) # (64,100)


        loss_ce = 1.0 * F.cross_entropy(logits_student_weak, target) + 1.0 * F.cross_entropy(logits_student_strong, target)
        loss_isv = 64.0 * isv_loss(logits_student_weak, logits_student_strong, logits_teacher_weak, logits_teacher_strong)
        loss_icv = 32.0 * icv_loss(logits_student_weak, logits_student_strong, logits_teacher_weak, logits_teacher_strong)

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_isv": loss_isv,
            "loss_icv": loss_icv,
        }
        values_dict = {
            "value_a": loss_isv,
            "value_b": loss_icv,
        }

        return logits_student_weak, losses_dict