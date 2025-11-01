import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from utils.options import args
import utils.common as utils
import numpy as np
import os
LearnedBatchNorm = nn.BatchNorm2d
N = args.N
M = args.M
I = args.iter
G = args.greedy_num
import math
class NonAffineBatchNorm(nn.BatchNorm2d):
    def __init__(self, dim):
        super(NonAffineBatchNorm, self).__init__(dim, affine=False)


DenseConv = nn.Conv2d

def get_n_m_sparse_matrix(w):
    length = w.numel()
    group = int(length / M)
    w_tmp = w.t().detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask = mask.scatter_(dim=1, index=index, value=0).reshape(w.t().shape).t()
    return w * mask, mask

def get_topk_sparse_matrix(w, ratio=0.5):
    """Get random sparse matrix using topk algorithm"""
    w_flat = w.flatten()
    num_elements = w_flat.numel()
    num_keep = int(num_elements * ratio)
    
    # Get topk indices based on absolute values
    _, topk_indices = torch.topk(w_flat.abs(), num_keep)
    
    # Create mask
    mask_flat = torch.zeros_like(w_flat)
    mask_flat[topk_indices] = 1
    
    # Reshape mask to match weight shape
    mask = mask_flat.reshape(w.shape)
    return w * mask, mask

def get_random_sparse_matrix(w, ratio=0.5):
    """Get random sparse matrix by randomly selecting elements to keep.
    
    Args:
        w (torch.Tensor): The input weight matrix.
        ratio (float): The ratio of elements to keep (sparsity ratio).
        
    Returns:
        tuple: (w_sparse, mask) where w_sparse is the sparse matrix and mask 
               is the binary mask.
    """
    w_flat = w.flatten()
    num_elements = w_flat.numel()
    num_keep = int(num_elements * ratio)
    
    # --- 核心修改部分：使用随机排列获取随机索引 ---
    # torch.randperm(n) 返回从 0 到 n-1 的一个随机排列
    # 选取前 num_keep 个作为要保留的元素的索引
    random_indices = torch.randperm(num_elements)[:num_keep]
    
    # 创建 mask
    mask_flat = torch.zeros_like(w_flat)
    # 将随机选中的索引位置设置为 1
    mask_flat[random_indices] = 1
    
    # Reshape mask to match weight shape
    mask = mask_flat.reshape(w.shape)
    
    # 应用 mask
    return w * mask, mask

def get_random_sparse_matrix_fast(w, ratio=0.5):
    """
    Get random sparse matrix by randomly selecting elements to keep.
    (Optimized version using rand + topk)
    
    Args:
        w (torch.Tensor): The input weight matrix.
        ratio (float): The ratio of elements to keep (sparsity ratio).
        
    Returns:
        tuple: (w_sparse, mask) where w_sparse is the sparse matrix and mask
               is the binary mask.
    """
    w_flat = w.flatten()
    num_elements = w_flat.numel()
    num_keep = int(num_elements * ratio)
    
    # --- 核心优化部分 ---
    # 1. 为每个元素生成一个随机分数
    #    确保随机分数张量与 w 在同一设备上 (e.g., 'cuda')
    random_scores = torch.rand_like(w_flat) 
    
    # 2. 使用 topk 找出得分最高的 num_keep 个索引
    #    这在 GPU 上的效率远高于 torch.randperm
    _, random_indices = torch.topk(random_scores, num_keep)
    # --- 优化结束 ---
    
    # 创建 mask
    mask_flat = torch.zeros_like(w_flat)
    
    # 将随机选中的索引位置设置为 1
    mask_flat[random_indices] = 1
    
    # Reshape mask to match weight shape
    mask = mask_flat.reshape(w.shape)
    
    # 应用 mask
    return w * mask, mask

def get_n_m_sparse_matrix_random(w, n=31, m=32):
    """
    实现 N:M 稀疏，但在 M 个元素的组内 *随机* 选择 N 个元素保留。
    
    注意：此实现遵循常见约定，即沿着权重张量的“输入”维度
    （在转置后是 dim=1）应用 N:M 稀疏。

    Args:
        w (torch.Tensor): 输入的权重张量。
        n (int): 每个 M 块中要保留的元素数量。
        m (int): 每个块的大小。

    Returns:
        tuple: (w_sparse, mask) 稀疏化后的权重和掩码。
    """
    # 1. 基本检查
    if n > m:
        raise ValueError(f"N ({n}) 不能大于 M ({m})")
    if w.numel() % m != 0:
        raise ValueError(f"张量总元素数 ({w.numel()}) "
                         f"必须能被 M ({m}) 整除")

    # 2. 重塑权重以应用分组稀疏
    # 遵循您参考函数中的约定：先转置，再重塑
    w_t = w.t()
    w_t_shape = w_t.shape
    num_groups = int(w_t.numel() / m)
    
    # 变形为 (num_groups, M)
    w_grouped = w_t.reshape(num_groups, m)

    # 3. 为每个元素生成随机分数
    # 形状：(num_groups, M)
    random_scores = torch.rand_like(w_grouped)

    # 4. 在每个组中（dim=1），找到分数最高的 N 个元素的索引
    # 这等同于在 M 个中随机选择 N 个
    # indices_to_keep 的形状：(num_groups, N)
    _, indices_to_keep = torch.topk(random_scores, n, dim=1)

    # 5. 创建掩码
    # 初始为全零
    mask_grouped = torch.zeros_like(w_grouped)
    
    # 将被选中的索引位置 1
    mask_grouped.scatter_(dim=1, index=indices_to_keep, value=1.0)

    # 6. 将掩码重塑回原始权重形状
    # 首先，重塑回转置后的形状
    mask_t = mask_grouped.reshape(w_t_shape)
    
    # 然后，转置回 w 的原始形状
    mask = mask_t.t()

    # 7. 应用掩码
    return w * mask, mask

class MyConv2d(autograd.Function):
    @staticmethod
    def forward(ctx, weight, inp_unf, forward_mask, backward_mask, decay = 0.0002):
        ctx.save_for_backward(weight, inp_unf, backward_mask)
        w_s = weight * forward_mask

        ctx.decay = decay
        ctx.mask = forward_mask

        out_unf = inp_unf.matmul(w_s)
        return out_unf

    @staticmethod
    def backward(ctx, g):
        weight, inp_unf, backward_mask = ctx.saved_tensors
        w_s = (weight * backward_mask).t()

        g_w_s = inp_unf.transpose(1,2).matmul(g).sum(0)
        g_w_s = g_w_s + ctx.decay * (1 - ctx.mask) * weight
        g_inp_unf = g.matmul(w_s)
        return g_w_s , g_inp_unf, None, None, None

class MyConv2d_Lay_m3(autograd.Function):
    @staticmethod
    def forward(ctx, weight, inp_unf, forward_mask, decay = 0.0002):
        ctx.save_for_backward(weight, inp_unf)
        w_s = weight * forward_mask

        ctx.decay = decay
        ctx.mask = forward_mask       

        out_unf = inp_unf.matmul(w_s)
        return out_unf

    @staticmethod
    def backward(ctx, g):
        
        weight, inp_unf = ctx.saved_tensors
        w_s = weight.t()

        g_w_s = inp_unf.transpose(1,2).matmul(g)
        g_w_s = g_w_s + ctx.decay * (1 - ctx.mask) * weight
        g_inp_unf = g.matmul(w_s)
        # g_b = g.sum(dim=1)

        return g_w_s , g_inp_unf, None, None


def get_best_permutation(w):
    length = w.numel()
    group = int(length / M)
    w = w.t()
    mask_sum_max = 0
    mask_sum_min = 1e8
    num_perm = G
    permutation = [v for v in range(w.size(0))]     
    p = np.random.permutation(permutation)        
    best_perm = permutation.copy()                  
    w_tmp = w.detach().abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask_f = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask_f = mask_f.scatter_(dim=1, index=index, value=0).reshape(w.shape)      

    for i in range(num_perm):
        mask = mask_f[p]
        w_t = w[p]
        # backward mask
        w_s = w_t * mask
        w_backward = w_s.t()
        
        w_tmp = w_backward.abs().reshape(group, M)
        index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
        mask_b = torch.ones(w_tmp.shape, device=w_tmp.device)
        mask_b = mask_b.scatter_(dim=1, index=index, value=0).reshape(w_backward.shape).t()
        final_mask = mask * mask_b
        if final_mask.sum() > mask_sum_max:
            mask_sum_max = final_mask.sum()
            # update thet best permutation
            best_perm = p
        # update permutation
        p = np.random.permutation(permutation)
    return best_perm



def get_n_m_backward_matrix(forward_mask, w_s, permutation):
    # import pdb; pdb.set_trace()
    w_s = w_s.t()
    forward_mask = forward_mask.t()
    length = w_s.numel()
    group = int(length / M)

    # change with the permutation
    mask = forward_mask[permutation]
    w_backward = w_s[permutation]
    w_backward = w_backward.t()
    
    # get the backwardmask
    w_tmp = w_backward.abs().reshape(group, M)
    index = torch.argsort(w_tmp, dim=1)[:, :int(M - N)]
    mask_b = torch.ones(w_tmp.shape, device=w_tmp.device)
    mask_b = mask_b.scatter_(dim=1, index=index, value=0).reshape(w_backward.shape).t()
    final_mask = mask * mask_b

    # recover the backwardmask with best permutation
    idx = torch.tensor(np.array([permutation])).t().repeat_interleave(w_s.size(1), dim=1)
    idx = idx.to(device=w_s.device)
    B_M = torch.zeros(w_s.size(), device=w_s.device).scatter_(dim=0, index=idx, src=final_mask).t()

    return B_M



# Bi-Mask
class NMConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.N = N  # number of non-zeros
        self.M = M
        self.flag = False
        self.iter = 0
        self.max_iter = I
        self.permute_idx = [v for v in range(self.weight.view(self.weight.size(0), -1).size(0))]
        self.device = self.weight.device
        self.forward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape, device=self.device).requires_grad_(False)
        self.backward_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape, device=self.device).requires_grad_(False)
        # self.grad_mask = torch.zeros(self.weight.view(self.weight.size(0), -1).t().shape, device=self.device).requires_grad_(False)
        self.unfold = nn.Unfold(kernel_size=self.kernel_size, dilation=self.dilation, padding=self.padding, stride=self.stride)
        
        # Add mask_mode support
        from utils.options import args
        self.mask_mode = getattr(args, 'mask_mode', 'm4')
        self.use_random_mask = getattr(args, 'use_random_mask', False)
        self.random_mask_ratio = getattr(args, 'random_mask_ratio', 0.5)

    def pre_mask_apply(self):
        if self.mask_mode == "m3":
            # Reshape weight to match forward_mask dimensions
            w = self.weight.view(self.weight.size(0), -1).t()
            if self.use_random_mask:
                w_s, _ = get_random_sparse_matrix_fast(w, self.random_mask_ratio)
            else:
                w_s, _ = get_n_m_sparse_matrix_random(w)
            self.weight.data = w_s.t().view(self.weight.shape)

    def grad_mask_apply(self):
        if self.mask_mode == "m5":
            w = self.weight.view(self.weight.size(0), -1).t()
            if self.use_random_mask:
                w_s, g_mask = get_random_sparse_matrix_fast(w, self.random_mask_ratio)
            else:
                w_s, g_mask = get_n_m_sparse_matrix_random(w)
            self.weight.data = w_s.t().view(self.weight.shape)
            g = self.weight.grad.view(self.weight.size(0), -1)
            g *= g_mask.t()
            
    def post_mask_apply(self):
        if self.mask_mode == "m4":
            # Reshape weight to match forward_mask dimensions
            w = self.weight.view(self.weight.size(0), -1).t()
            if self.use_random_mask:
                w_s, _ = get_random_sparse_matrix_fast(w, self.random_mask_ratio)
            else:
                w_s, _ = get_n_m_sparse_matrix_random(w)
            self.weight.data = w_s.t().view(self.weight.shape)
        

    def forward(self, x):
        # import pdb;pdb.set_trace()
        
        w = self.weight.view(self.weight.size(0), -1).t()
        
        # Choose mask type based on configuration
        if self.use_random_mask:
            # Use random mask with topk algorithm
            w_s, self.forward_mask = get_random_sparse_matrix_fast(w, self.random_mask_ratio)
        else:
            # Use N:M semi-structured mask
            if self.iter % self.max_iter == 0:
                self.permute_idx = get_best_permutation(w)
            w_s, self.forward_mask = get_n_m_sparse_matrix_random(w)
        
        # Apply different mask modes
        if self.mask_mode == "m2":
            # Bidirectional mask mode
            if not self.use_random_mask:
                # self.backward_mask = get_n_m_sparse_matrix_random(self.forward_mask, w_s, self.permute_idx)   
                self.backward_mask = self.forward_mask
            else:
                self.backward_mask = self.forward_mask
            inp_unf = self.unfold(x)
            out_unf = MyConv2d.apply(w, inp_unf.transpose(1, 2), self.forward_mask, self.backward_mask)
        # elif self.mask_mode == "m3" or self.mask_mode == "m4":
        else:
            # Forward mask only mode
            inp_unf = self.unfold(x)
            out_unf = MyConv2d_Lay_m3.apply(w, inp_unf.transpose(1, 2), self.forward_mask)

            if self.mask_mode == "m3":
                # import pdb; pdb.set_trace()
                # w_view = self.weight.data.view(self.weight.size(0), -1)
                # w_view *= self.forward_mask.t()
                pass

        if self.flag == False:
            self.fold = nn.Fold((int(math.sqrt(out_unf.shape[1])), int(math.sqrt(out_unf.shape[1]))), (1,1))
            self.flag = True
        out = self.fold(out_unf.transpose(1, 2))
        return out
