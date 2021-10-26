import torch
import torch.nn as nn
from loss.emd import emd_module as emd
from loss.chamfer.champfer_loss import ChamferLoss
from utils.pointnet_utils import gather_points, group_points, farthest_point_sample
from typing import Optional

def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

def calc_cd(output, gt, calc_f1=False):
    chamfer_loss = ChamferLoss()
    dist1, dist2 = chamfer_loss(output, gt)
    cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
    cd_t = (dist1.mean(1) + dist2.mean(1))
    if calc_f1:
        f1, _, _ = fscore(dist1, dist2)
        return cd_p, cd_t, f1
    else:
        return cd_p, cd_t

def calc_emd(output, gt, eps=0.005, iterations=50):
    emd_loss = emd.emdModule()
    dist, matches = emd_loss(output, gt, eps, iterations)
    emd_out = torch.sqrt(dist).mean(1)
    return emd_out

def knn_point(point_ref, point_query, k):
    """
    :param point_ref:   [B,N,C]
    :param point_query: [B,M,C]
    :param k: the number of neighbor points
    :return: dist, idx ->  [B,M,K], [B,M,K]
    """
    dists = []
    idxs = []
    for ref, query in zip(point_ref, point_query):
        n = ref.shape[0]
        m = query.shape[0]
        inner = -2 * torch.matmul(query, ref.T)
        xx = torch.sum(query ** 2, dim=1, keepdim=True).repeat(1, n)
        yy = torch.sum(ref ** 2, dim=1, keepdim=False).unsqueeze(0).repeat(m, 1)
        pairwise_distance = -xx - inner - yy
        dist, idx = pairwise_distance.topk(k=k, dim=-1)
        dists.append(dist)
        idxs.append(idx)
    dist, idx = torch.stack(dists, dim=0), torch.stack(idxs, dim=0)
    return dist, idx

def MLP(channels, batch_norm = False, conv_type = '1D', act:Optional=nn.ReLU, last_act=True):
    conv_bn = {'1D':[nn.Conv1d, nn.BatchNorm1d, (1,)],
               '2D':[nn.Conv2d, nn.BatchNorm2d, (1,1)],
               'linear':[nn.Linear, nn.BatchNorm1d, None]}

    conv, bn, kernel = conv_bn[conv_type]

    n = len(channels)
    nn_list = []
    for i in range(1, n):
        if kernel:
            nn_list.append(conv(channels[i-1], channels[i], kernel))
        else:
            nn_list.append(conv(channels[i-1], channels[i]))

        if i!=n-1 or last_act:
            if batch_norm:
                nn_list.append(bn(channels[i]))
            nn_list.append(act(inplace=True))

    return nn.Sequential(*nn_list)

def fps(xyz, npoints, BNC=True):
    if BNC:
        idx = farthest_point_sample(xyz, npoints)
        xyz_new = gather_points(xyz.transpose(2, 1).contiguous(), idx)  # (B, 3, npoints)
        xyz_new = xyz_new.transpose(2, 1).contiguous()  # (B, npoints, 3)
    else:
        idx = farthest_point_sample(xyz.transpose(2, 1).contiguous(), npoints)
        xyz_new = gather_points(xyz, idx)  # (B, 3, npoints)

    return xyz_new

def cov(xyz_ref, xyz_query=None, k=8, index=False, idx = None):
    """
    Args:
        xyz: (B, N, 3)
        k: the number of neighbor points
    Returns:
            (B, 9, N)
    """
    B, _, C = xyz_ref.shape
    if idx is None and xyz_query is None:
        xyz_query = xyz_ref
        _, idx = knn_point(xyz_ref, xyz_query, k)

    x_knn = group_points(xyz_ref.transpose(2,1).contiguous(), idx).permute(0, 2, 3, 1)  # (B, 3, N, k)  -> (B, N, K, 3)
    x_knn = x_knn - x_knn.mean(dim=-2, keepdim=True)
    _cov = torch.matmul(x_knn.transpose(3, 2), x_knn) / (k - 1)  # (B, N, 3, 3)
    if index:
        return _cov.view(B, -1, 9).transpose(2,1).contiguous(), idx  # (B, 9, N)
    else:
        return _cov.view(B, -1, 9).transpose(2, 1).contiguous()