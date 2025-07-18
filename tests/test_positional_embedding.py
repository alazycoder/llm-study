import pytest
import torch
from llm import TraditionalRoPE


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("seq_length", [128, 256, 512])
@pytest.mark.parametrize("head_num", [1, 8])
@pytest.mark.parametrize("embed_dim", [128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32])
def test_traditional_rope(batch_size, seq_length, head_num, embed_dim, dtype):
    x = torch.randn(batch_size, seq_length, embed_dim, dtype=dtype, device='cuda')
    traditional_rope_module = TraditionalRoPE(head_num, embed_dim, dtype)
    mask = torch.triu(torch.ones(seq_length, seq_length, dtype=dtype, device='cuda')) * -1e9 # (seq_len, seq_len)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    output = traditional_rope_module(x, mask)
    assert output.shape == (batch_size, seq_length, embed_dim)
