import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim, dtype):
        super().__init__()
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, embed_dim, dtype=dtype, device='cuda')
        self.w_k = nn.Linear(embed_dim, embed_dim, dtype=dtype, device='cuda')
        self.w_v = nn.Linear(embed_dim, embed_dim, dtype=dtype, device='cuda')
    
    def forward(self, x, mask=None):
        # x: [BATCH, SEQ_LEN, EMBED_DIM]
        # mask: [BATCH, SEQ_LEN, SEQ_LEN]
        Q = self.w_q(x) # [BATCH, SEQ_LEN, EMBED_DIM]
        K = self.w_k(x) # [BATCH, SEQ_LEN, EMBED_DIM]
        V = self.w_v(x) # [BATCH, SEQ_LEN, EMBED_DIM]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        # scores: [BATCH, SEQ_LEN, SEQ_LEN]
        if mask is not None:
            scores = scores + mask # mask: 0 or -inf
        
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output


class SimpleMultiHeadAttention(nn.Module):
    def __init__(self, head_num, embed_dim, dtype):
        super().__init__()
        self.head_num = head_num
        self.embed_dim = embed_dim
        self.w_q = nn.Linear(embed_dim, head_num * embed_dim, dtype=dtype, device='cuda')
        self.w_k = nn.Linear(embed_dim, head_num * embed_dim, dtype=dtype, device='cuda')
        self.w_v = nn.Linear(embed_dim, head_num * embed_dim, dtype=dtype, device='cuda')
        self.w_o = nn.Linear(head_num * embed_dim, embed_dim, dtype=dtype, device='cuda')
    
    def forward(self, x, mask=None):
        # x: [BATCH, SEQ_LEN, EMBED_DIM]
        # mask: [BATCH, SEQ_LEN, SEQ_LEN]
        Q = self.w_q(x) # [BATCH, SEQ_LEN, HEAD_NUM * EMBED_DIM]
        K = self.w_k(x) # [BATCH, SEQ_LEN, HEAD_NUM * EMBED_DIM]
        V = self.w_v(x) # [BATCH, SEQ_LEN, HEAD_NUM * EMBED_DIM]
        
        batch_size = x.shape[0]
        Q = Q.view(batch_size, -1, self.head_num, self.embed_dim).transpose(1, 2)
        # [BATCH, HEAD_NUM, SEQ_LEN, EMBED_DIM]
        K = K.view(batch_size, -1, self.head_num, self.embed_dim).transpose(1, 2)
        # [BATCH, HEAD_NUM, SEQ_LEN, EMBED_DIM]
        V = V.view(batch_size, -1, self.head_num, self.embed_dim).transpose(1, 2)
        # [BATCH, HEAD_NUM, SEQ_LEN, EMBED_DIM]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        # [BATCH, HEAD_NUM, SEQ_LEN, SEQ_LEN]

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)
            scores = scores + mask # mask: 0 or -inf
        
        attn_weights = F.softmax(scores, dim=-1) # [BATCH, HEAD_NUM, SEQ_LEN, SEQ_LEN]
        attn = torch.matmul(attn_weights, V) # [BATCH, HEAD_NUM, SEQ_LEN, EMBED_DIM]
        output = self.w_o(attn.transpose(1, 2).contiguous().view(batch_size, -1, self.head_num * self.embed_dim))
        return output
