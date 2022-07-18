import logging
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

def _cost_matrix_cosine(x, y, eps=1e-5):
    """Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))# torch.size([batch_size, txt_token, img_token])
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def _trace(x):
    """compute trace of input tensor (batched)"""
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def _ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """[B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device) / x_len.unsqueeze(1)# 如果除0是inf
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2) / beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)# ??torch.size([batch_size, tokens])
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def _compute_ot_loss(img_emb, txt_emb, img_mask, txt_mask, itm_labels):
    assert img_emb.size(0) == txt_emb.size(0)

    with torch.cuda.amp.autocast(enabled=False):
        txt_mask, img_mask = txt_mask.bool(), img_mask.bool()# torch.size([batch_size, tokens])
        for i, _len in enumerate(txt_mask.sum(dim=1)):
            txt_mask[i, _len - 1] = False# SEP token对应置false
        txt_mask[:, 0] = False# CLS token对应置false
        img_mask[:, 0] = False# ?image也有特殊符号吗

        txt_pad, img_pad = ~txt_mask, ~img_mask

        cost = _cost_matrix_cosine(txt_emb.float(), img_emb.float())# torch.size([batch_size, txt_token, img_token])
        joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)#图文俩组token任一mask的位置都mask掉 torch.size([batch_size, txt_token, img_token] = torch.size([batch_size, txt_token, 1]) | torch.size([batch_size, 1, img_token]
        cost.masked_fill_(joint_pad, 0) # joint_pad为true的地方指示cost相应位置置为0，# torch.size([batch_size, txt_token, img_token])

        txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)).to(
            dtype=cost.dtype
        )
        T = _ipot(
            cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad, 0.5, 50, 1
        )
        distance = _trace(cost.matmul(T.detach()))

    dist_pos = distance.masked_select(itm_labels == 1)
    dist_neg = distance.masked_select(itm_labels == 0)
    ot_loss = (dist_pos.sum() - dist_neg.sum()) / (dist_pos.size(0) + dist_neg.size(0))

    return ot_loss

if __name__ == '__main__':
    print("测试optimal transport")
    batch_size = 1
    txt_token, img_token = 2, 2
    hidden_dim = 5
    img_emb = torch.randn(batch_size, img_token, hidden_dim)
    txt_emb = torch.randn(batch_size, txt_token, hidden_dim)
    txt_mask = torch.ones(batch_size, txt_token)
    img_mask = torch.ones(batch_size, img_token)
    itm_labels = torch.ones(batch_size)
    _compute_ot_loss(img_emb, txt_emb, img_mask, txt_mask, itm_labels)