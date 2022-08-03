import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import glob
import json
import tqdm
import functools

def cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]
    [L_x, L_y]对应每个txt 和image 的token的cosine相似度距离"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x) # [b, m, m]
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace # [b]


@torch.no_grad()
def ipot(C       , x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B]  , [B, M], [B], [B, N], [B, M, N], 0.5,  50,        1"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)# [B, M] / [B, M] = [B, M]txt的每个token位置1/len（len是除了特殊字符的有效长度）
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta) # e^(-cost)距离越大值越小 [B, N, M]

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)# [B, 1, 1]
    y_len = y_len.unsqueeze(1).unsqueeze(2)# [B, 1, 1]

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1) # [B, 1, M] 值为1e4或者0(masked部分为1e4)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1) # [B, 1, N]

    for _ in range(iteration):
        Q = A * T  # [B, N, M] * [B, N, M] = [B, N, M]
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)# 1/([b, 1, 1] * (([b, n, m] * [b, m, 1])->[b, 1, n]) + [b, 1, n]) = [b, 1, n]
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)# [b, 1, m]
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0) # [b, n, m]
    return T


def optimal_transport_dist(
    txt_emb, img_emb, txt_pad, img_pad, beta=0.5, iteration=50, k=1
):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(dtype=cost.dtype)

    T = ipot(
        cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, beta, iteration, k
    )
    distance = trace(cost.matmul(T.detach()))
    return distance

if __name__ == '__main__':
    with torch.cuda.amp.autocast(enabled=False):
        batch_size = 2
        L_x = 2
        L_y = 4
        hidden_size = 10
        txt_emb, img_emb = torch.randn([batch_size,L_x,hidden_size]), torch.randn([batch_size,L_y,hidden_size])
        txt_mask, img_mask = torch.ones([batch_size,L_x]).bool(), torch.ones([batch_size,L_y]).bool()
        itm_labels = torch.ones([batch_size])
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False
        txt_mask[:, 0] = False
        img_mask[:, 0] = False
        # if "deit" in pl_module.hparams.config["vit"]:
        #     img_mask[:, 1] = False
        txt_pad, img_pad = ~txt_mask, ~img_mask # pad: [B, L_x] [B, L_y] set True  if masked else False

        cost = cost_matrix_cosine(txt_emb.float(), img_emb.float())
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2) # [B, L_x, L_y] 
        cost.masked_fill_(joint_pad, 0) # txt和image对应token mask掉的cost置0

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        ) # [b, L_y, L_x]
        distance = trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1) # positive正例的distance
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))