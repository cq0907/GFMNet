import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable
from aligned.local_dist import *
from IPython import embed
from re_rank import random_walk, k_reciprocal

class OriTripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """
    
    def __init__(self, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct    
        
# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6 # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative  = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


class TripletLoss_ADP0(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha=1, gamma=1, square=0, margin=0.3):
        super(TripletLoss_ADP0, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        # self.ranking_loss_global = nn.MarginRankingLoss(margin=margin)
        self.alpha = alpha  # 1
        self.gamma = gamma  # 1
        self.square = square  # 1

    def forward(self, inputs, targets, local_features=None, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        dist_ap4, dist_an4, p_inds4, n_inds4 = hard_example_mining(dist_mat, targets, return_inds=True)
        local_features = local_features.permute(0, 2, 1)
        p_local_features = local_features[p_inds4]
        n_local_features = local_features[n_inds4]
        local_dist_ap = batch_local_dist1(local_features, p_local_features, dist_an4)
        local_dist_an = batch_local_dist1(local_features, n_local_features, dist_an4)
        # Compute ranking hinge loss
        yy = torch.ones_like(dist_an4)
        local_loss = self.ranking_loss_local(local_dist_an, local_dist_ap, yy)
        # global_loss = self.ranking_loss_global(dist_an4, dist_ap4, yy)
        # =============================================================================

        # squared difference
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            # diff_pow = torch.clamp_max(diff_pow, max=88)
            diff_pow = torch.clamp_max(diff_pow, max=44)
            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            loss = self.ranking_loss(diff_pow, y)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, local_loss, correct

class TripletLoss_ADP(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha =1, gamma = 1, square = 0, margin=0.3):
        super(TripletLoss_ADP, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        # self.ranking_loss_global = nn.MarginRankingLoss(margin=margin)
        self.alpha = alpha  # 1
        self.gamma = gamma  # 1
        self.square = square  # 1

    def forward(self, inputs, targets, local_features=None, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap*self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an*self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        
        # ranking_loss = nn.SoftMarginLoss(reduction = 'none')
        # loss1 = ranking_loss(closest_negative - furthest_positive, y)

        # =======================================================
        # For each anchor, find the hardest positive and negative
        # n = inputs.shape[0] // 3
        # feat1 = inputs.narrow(0, 0, n)
        # feat2 = inputs.narrow(0, n, n)
        # feat3 = inputs.narrow(0, 2 * n, n)
        # target1 = targets.narrow(0, 0, n)
        # target2 = targets.narrow(0, n, n)
        # target3 = targets.narrow(0, 2 * n, n)
        # lf1 = local_features.narrow(0, 0, n)
        # lf2 = local_features.narrow(0, n, n)
        # lf3 = local_features.narrow(0, 2 * n, n)
        #
        # dist_mat1 = pdist_torch(feat1, feat1)
        # dist_mat2 = pdist_torch(feat2, feat2)
        # dist_mat3 = pdist_torch(feat3, feat3)
        #
        # dist_ap1, dist_an1, p_inds1, n_inds1 = hard_example_mining(dist_mat1, target1, return_inds=True)
        # lf1 = lf1.permute(0, 2, 1)
        # p_lf1 = lf1[p_inds1]
        # n_lf1 = lf1[n_inds1]
        # lf1_dist_ap = batch_local_dist(lf1, p_lf1)
        # lf1_dist_an = batch_local_dist(lf1, n_lf1)
        # # Compute ranking hinge loss
        # yy = torch.ones_like(dist_an1)
        # local_loss1 = self.ranking_loss_local(lf1_dist_an, lf1_dist_ap, yy)
        #
        # dist_ap2, dist_an2, p_inds2, n_inds2 = hard_example_mining(dist_mat2, target2, return_inds=True)
        # lf2 = lf2.permute(0, 2, 1)
        # p_lf2 = lf2[p_inds2]
        # n_lf2 = lf2[n_inds2]
        # lf2_dist_ap = batch_local_dist(lf2, p_lf2)
        # lf2_dist_an = batch_local_dist(lf2, n_lf2)
        # # Compute ranking hinge loss
        # yy = torch.ones_like(dist_an2)
        # local_loss2 = self.ranking_loss_local(lf2_dist_an, lf2_dist_ap, yy)
        #
        # dist_ap3, dist_an3, p_inds3, n_inds3 = hard_example_mining(dist_mat3, target3, return_inds=True)
        # lf3 = lf3.permute(0, 2, 1)
        # p_lf3 = lf3[p_inds3]
        # n_lf3 = lf3[n_inds3]
        # lf3_dist_ap = batch_local_dist(lf3, p_lf3)
        # lf3_dist_an = batch_local_dist(lf3, n_lf3)
        # # Compute ranking hinge loss
        # yy = torch.ones_like(dist_an3)
        # local_loss3 = self.ranking_loss_local(lf3_dist_an, lf3_dist_ap, yy)
        #
        # # local_loss = local_loss1 + local_loss2 + local_loss3

        dist_ap4, dist_an4, p_inds4, n_inds4 = hard_example_mining(dist_mat, targets, return_inds=True)
        local_features = local_features.permute(0, 2, 1)
        p_local_features = local_features[p_inds4]
        n_local_features = local_features[n_inds4]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)
        # Compute ranking hinge loss
        yy = torch.ones_like(dist_an4)
        local_loss = self.ranking_loss_local(local_dist_an, local_dist_ap, yy)
        # global_loss = self.ranking_loss_global(dist_an4, dist_ap4, yy)
        # =============================================================================

        # squared difference
        if self.square ==0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            # diff_pow = torch.clamp_max(diff_pow, max=88)
            diff_pow = torch.clamp_max(diff_pow, max=44)
            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            loss = self.ranking_loss(diff_pow, y)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, local_loss, correct

class TripletLoss_ADP1(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha=1, gamma=1, square=0, margin=0.3):
        super(TripletLoss_ADP1, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        # self.ranking_loss_global = nn.MarginRankingLoss(margin=margin)
        self.alpha = alpha  # 1
        self.gamma = gamma  # 1
        self.square = square  # 1

    def forward(self, inputs, targets, local_features=None, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        label_uni = targets.unique()
        label_num = len(label_uni)
        target = torch.cat([label_uni, label_uni, label_uni])
        feat = inputs.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        center = torch.cat(center)
        n = center.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(center, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, center, center.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        is_pos = target.expand(n, n).eq(target.expand(n, n).t())
        is_neg = target.expand(n, n).ne(target.expand(n, n).t())
        dist_ap = dist * is_pos
        dist_an = dist * is_neg
        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)


        dist_ap4, dist_an4, p_inds4, n_inds4 = hard_example_mining(dist_mat, targets, return_inds=True)
        local_features = local_features.permute(0, 2, 1)
        p_local_features = local_features[p_inds4]
        n_local_features = local_features[n_inds4]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)
        # Compute ranking hinge loss
        yy = torch.ones_like(dist_an4)
        local_loss = self.ranking_loss_local(local_dist_an, local_dist_ap, yy)
        # global_loss = self.ranking_loss_global(dist_an4, dist_ap4, yy)
        # =============================================================================

        # squared difference
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)
            # diff_pow = torch.clamp_max(diff_pow, max=44)
            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            loss = self.ranking_loss(diff_pow, y)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, local_loss, correct

class TripletLoss_ADP2(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self, alpha=1, gamma=1, square=0, margin=0.3):
        super(TripletLoss_ADP2, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.alpha = alpha  # 1
        self.gamma = gamma  # 1
        self.square = square  # 1

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap * self.alpha, is_pos)
        weights_an = softmax_weights(-dist_an * self.alpha, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        # squared difference
        if self.square == 0:
            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss(self.gamma * (closest_negative - furthest_positive), y)
        else:
            diff_pow = torch.pow(furthest_positive - closest_negative, 2) * self.gamma
            diff_pow = torch.clamp_max(diff_pow, max=88)
            # diff_pow = torch.clamp_max(diff_pow, max=44)
            # Compute ranking hinge loss
            y1 = (furthest_positive > closest_negative).float()
            y2 = y1 - 1
            y = -(y1 + y2)
            loss = self.ranking_loss(diff_pow, y)

        # loss = self.ranking_loss(self.gamma*(closest_negative - furthest_positive), y)
        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

class TripletLossAlignedReID(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLossAlignedReID, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets, local_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        dist_ap,dist_an,p_inds,n_inds = hard_example_mining(dist,targets,return_inds=True)
        local_features = local_features.permute(0,2,1)
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        local_loss = self.ranking_loss_local(local_dist_an,local_dist_ap, y)
        if self.mutual:
            return global_loss+local_loss,dist
        return global_loss,local_loss

class TripletLossAlignedReID1(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLossAlignedReID1, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, feats, labels, local_features):
        """
        Args:
            feats: feature matrix with shape (batch_size, feat_dim)
            labels: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        label_num = len(label_uni)
        targets = torch.cat([label_uni, label_uni, label_uni])
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))

        inputs = torch.cat(center)

        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        dist_ap,dist_an,p_inds,n_inds = hard_example_mining(dist,targets,return_inds=True)
        local_features = local_features.permute(0,2,1)
        p_local_features = inputs[p_inds]
        n_local_features = inputs[n_inds]
        inputs=inputs.reshape((n,6,-1))
        p_local_features = p_local_features.reshape((n,6,-1))
        n_local_features = n_local_features.reshape((n, 6, -1))
        local_dist_ap = batch_local_dist(inputs, p_local_features)
        local_dist_an = batch_local_dist(inputs, n_local_features)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        local_loss = self.ranking_loss_local(local_dist_an,local_dist_ap, y)
        return global_loss+local_loss

class TripletLossAlignedReID2(nn.Module):
    """Triplet loss with hard positive/negative mining.

    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.

    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.

    Args:
        margin (float): margin for triplet.
    """
    def __init__(self, margin=0.3, mutual_flag = False):
        super(TripletLossAlignedReID2, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.ranking_loss_local = nn.MarginRankingLoss(margin=margin)
        self.mutual = mutual_flag

    def forward(self, inputs, targets, local_features):
        """
        Args:
            inputs: feature matrix with shape (batch_size, feat_dim)
            targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)
        #inputs = 1. * inputs / (torch.norm(inputs, 2, dim=-1, keepdim=True).expand_as(inputs) + 1e-12)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        # For each anchor, find the hardest positive and negative
        dist_ap,dist_an,p_inds,n_inds = hard_example_mining(dist,targets,return_inds=True)
        local_features = local_features.permute(0,2,1)
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]
        local_dist_ap = batch_local_dist(local_features, p_local_features)
        local_dist_an = batch_local_dist(local_features, n_local_features)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        global_loss = self.ranking_loss(dist_an, dist_ap, y)
        local_loss = self.ranking_loss_local(local_dist_an,local_dist_ap, y)
        if self.mutual:
            return global_loss+local_loss,dist
        return global_loss+local_loss

class CenterTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))

        inputs = torch.cat(center)

        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class CenterAlignedTripletLoss(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterAlignedTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels, local_features):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))

        inputs = torch.cat(center)

        m = labels.size(0)
        dist = pdist_torch(inputs, feats)
        is_pos = labels.expand(m, m).eq(labels.expand(m, m).t())
        is_neg = labels.expand(m, m).ne(labels.expand(m, m).t())

        is_pos = is_pos.chunk(label_num * 3, 0)
        is_neg = is_neg.chunk(label_num * 3, 0)

        is_pos = [torch.unsqueeze(p[0], dim=0) for p in is_pos]
        is_neg = [torch.unsqueeze(n[0], dim=0) for n in is_neg]
        is_pos = torch.cat(is_pos, dim=0)
        is_neg = torch.cat(is_neg, dim=0)

        n = dist.size(0)
        dist_ap, relative_p_inds = torch.max(dist[is_pos].contiguous().view(n, -1), 1, keepdim=True)
        dist_an, relative_n_inds = torch.min(dist[is_neg].contiguous().view(n, -1), 1, keepdim=True)
        dist_ap = dist_ap.squeeze(1)
        dist_an = dist_an.squeeze(1)

        ind = (labels.new().resize_as_(labels).copy_(torch.arange(0, m).long()).unsqueeze(0).expand(n, m))
        # shape [N, 1]
        p_inds = torch.gather(ind[is_pos].contiguous().view(n, -1), 1, relative_p_inds.data)
        n_inds = torch.gather(ind[is_neg].contiguous().view(n, -1), 1, relative_n_inds.data)
        # shape [N]
        p_inds = p_inds.squeeze(1)
        n_inds = n_inds.squeeze(1)

        local_features = local_features.permute(0, 2, 1)
        p_local_features = local_features[p_inds]
        n_local_features = local_features[n_inds]

        inputs1 = torch.reshape(inputs, (p_local_features.shape[0], p_local_features.shape[1], -1))
        local_dist_ap = batch_local_dist(inputs1, p_local_features)
        local_dist_an = batch_local_dist(inputs1, n_local_features)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_ap)
        loss = self.ranking_loss(local_dist_an, local_dist_ap, y)

        return loss

class CenterAlignedTripletLoss1(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterAlignedTripletLoss1, self).__init__()
        # self.ranking_loss = nn.SoftMarginLoss()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        label_uni = labels.unique()
        label_num = len(label_uni)
        targets = torch.cat([label_uni, label_uni, label_uni])
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))

        inputs = torch.cat(center)

        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

        dist_ap = dist * is_pos
        dist_an = dist * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)

        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        # y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        # loss = self.ranking_loss(closest_negative - furthest_positive, y)

        y = torch.ones_like(closest_negative)
        loss = self.ranking_loss(closest_negative, furthest_positive, y)

        # correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss

class CenterAlignedTripletLoss2(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, alpha=1, gamma=1, square=0, margin=0.3):
        super(CenterAlignedTripletLoss2, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        label_uni = labels.unique()
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 3, 0)
        intra = []
        for i in range(label_num * 3):
            feat_i = feat[i]
            center_i = torch.mean(feat[i], dim=0, keepdim=True)
            dist_i = pdist_torch(feat_i, center_i)
            max_v_i = torch.max(dist_i, dim=0, keepdim=True)[0]
            diff_i = dist_i - max_v_i
            Z_i = torch.sum(torch.exp(diff_i), dim=0, keepdim=True) + 1e-6
            W_i = torch.exp(diff_i) / Z_i

            intra.append(torch.mean(dist_i * W_i, dim=0))

        intra = torch.cat(intra)
        y = intra.new().resize_as_(intra).fill_(-1)
        loss = self.ranking_loss(intra, y)

        return loss

class CenterAlignedTripletLoss3(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterAlignedTripletLoss3, self).__init__()
        self.ranking_loss0 = nn.SoftMarginLoss()
        self.ranking_loss1 = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels, flag=0):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        label_uni = labels.unique()
        label_num = len(label_uni)
        targets = torch.cat([label_uni, label_uni, label_uni])
        feat = feats.chunk(label_num * 3, 0)
        center = []
        for i in range(label_num * 3):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))

        inputs = torch.cat(center)

        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        if flag == 0:

            is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
            is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

            dist_ap = dist * is_pos
            dist_an = dist * is_neg

            weights_ap = softmax_weights(dist_ap, is_pos)
            weights_an = softmax_weights(-dist_an, is_neg)

            furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
            closest_negative = torch.sum(dist_an * weights_an, dim=1)

            y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
            loss = self.ranking_loss0(closest_negative - furthest_positive, y)
            correct = torch.ge(closest_negative, furthest_positive).sum().item()
        else:
            mask = targets.expand(n, n).eq(targets.expand(n, n).t())
            dist_ap, dist_an = [], []
            for i in range(n):
                dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
            dist_ap = torch.cat(dist_ap)
            dist_an = torch.cat(dist_an)
            # Compute ranking hinge loss
            y = torch.ones_like(dist_an)
            loss = self.ranking_loss1(dist_an, dist_ap, y)
            # compute accuracy
            correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct

class CenterAlignedTripletLoss4(nn.Module):
    """ Hetero-center-triplet-loss-for-VT-Re-ID
   "Parameters Sharing Exploration and Hetero-Center Triplet Loss for Visible-Thermal Person Re-Identification"
   [(arxiv)](https://arxiv.org/abs/2008.06223).

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(CenterAlignedTripletLoss4, self).__init__()
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """

        label_uni = labels.unique()
        label_num = len(label_uni)
        targets = torch.cat([label_uni, label_uni, label_uni])
        feat = feats.chunk(label_num * 3, 0)
        center = []
        intra = []
        for i in range(label_num * 3):
            feat_i = feat[i]
            dist_i = pdist_torch(feat_i, feat_i)
            max_v_i = torch.max(dist_i, dim=0, keepdim=True)[0]
            diff_i = dist_i - max_v_i

            intra.append(torch.mean(torch.exp(diff_i)).unsqueeze(0))
            center.append(torch.mean(feat[i], dim=0, keepdim=True))

        inputs = torch.cat(center)
        intra = torch.cat(intra)

        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss + torch.mean(intra)*0.1, correct

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class KLDivLoss(nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number
        T=3
        predict = F.log_softmax(pred/T,dim=1)
        target_data = F.softmax(label/T,dim=1)
        target_data =target_data+10**(-7)
        target = Variable(target_data.data.cuda(),requires_grad=False)
        loss=T*T*((target*(target.log()-predict)).sum(1).sum()/target.size()[0])
        return loss

class JSDivLoss(nn.Module):
    def __init__(self):
        super(JSDivLoss, self).__init__()
    def forward(self, pred, label):
        # pred: 2D matrix (batch_size, num_classes)
        # label: 1D vector indicating class number

        predict = F.softmax(pred, dim=1)
        target = F.softmax(label, dim=1)
        # target = target+10**(-7)
        # target = Variable(target.data.cuda(), requires_grad=False)
        temp = (predict + target) / 2.0
        loss = 0.5*((target * (target.log() - temp.log())).sum(1).sum() / target.size()[0]) + 0.5*((predict * (predict.log() - temp.log())).sum(1).sum() / predict.size()[0])
        return loss
       
def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx    


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis = 1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis = 1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx


