import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import SimpleMultiHeadAttention


MAX_SEQ_LEN = 512


class TraditionalRoPE(SimpleMultiHeadAttention):
    def __init__(self, head_num, embed_dim, dtype):
        if embed_dim % 2 != 0:
            raise ValueError(f"embed_dim should be even, but get {embed_dim}")
        super().__init__(head_num, embed_dim, dtype)
        # theta = 1 / (10000^(2j/d)) where j in [0, embed_dim//2]
        theta = 1 / (10000.0 ** (torch.arange(0, embed_dim, 2, dtype=dtype, device='cuda') / embed_dim))
        # angle = m * theta where m in range[0, MAX_SEQ_LEN)
        angle = torch.outer(torch.arange(MAX_SEQ_LEN, dtype=dtype, device='cuda'), theta)
        # angle [MAX_SEQ_LEN, embed_dim//2]
        self.freqs = torch.polar(torch.ones_like(angle), angle)
        # torch.polar(abs, angle) -> abs * cos(angle) + abs * sin(angle) * i  where i^2=-1
        # self.freqs dtype=torch.complex

    def forward(self, x, mask=None):
        # x: [BATCH, SEQ_LEN, EMBED_DIM]
        # mask: [BATCH, SEQ_LEN, SEQ_LEN]
        Q = self.w_q(x) # [BATCH, SEQ_LEN, HEAD_NUM * EMBED_DIM]
        K = self.w_k(x) # [BATCH, SEQ_LEN, HEAD_NUM * EMBED_DIM]
        V = self.w_v(x) # [BATCH, SEQ_LEN, HEAD_NUM * EMBED_DIM]

        batch_size, seq_len = x.shape[0], x.shape[1]
        Q = torch.view_as_complex(Q.view(batch_size, seq_len, self.head_num, self.embed_dim//2, 2).transpose(1, 2))
        K = torch.view_as_complex(K.view(batch_size, seq_len, self.head_num, self.embed_dim//2, 2).transpose(1, 2))
        V = V.view(batch_size, seq_len, self.head_num, self.embed_dim).transpose(1, 2)

        freq = self.freqs[:seq_len, :].unsqueeze(0).unsqueeze(1).repeat(batch_size, self.head_num, 1, 1)
        Q_ = torch.view_as_real(Q * freq).flatten(3)
        K_ = torch.view_as_real(K * freq).flatten(3)

        scores = torch.matmul(Q_, K_.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.head_num, 1, 1)
            scores = scores + mask # mask: 0 or -inf
        
        attn_weights = F.softmax(scores, dim=-1) # [BATCH, HEAD_NUM, SEQ_LEN, SEQ_LEN]
        attn = torch.matmul(attn_weights, V) # [BATCH, HEAD_NUM, SEQ_LEN, EMBED_DIM]
        output = self.w_o(attn.transpose(1, 2).contiguous().view(batch_size, seq_len, self.head_num * self.embed_dim))
        return output
