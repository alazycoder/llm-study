import pytest
import torch
from llm import SimpleSelfAttention, SimpleMultiHeadAttention


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("seq_length", [128, 512, 1024])
@pytest.mark.parametrize("embed_dim", [128, 512, 1024])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_simple_self_attention(batch_size, seq_length, embed_dim, dtype):
    x = torch.randn(batch_size, seq_length, embed_dim, dtype=dtype, device='cuda')
    self_attention_module = SimpleSelfAttention(embed_dim, dtype)
    mask = torch.triu(torch.ones(seq_length, seq_length, dtype=dtype, device='cuda')) * -1e9 # (seq_len, seq_len)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    output = self_attention_module(x, mask)
    assert output.shape == (batch_size, seq_length, embed_dim)


@pytest.mark.parametrize("batch_size", [1, 128])
@pytest.mark.parametrize("seq_length", [128, 256, 512])
@pytest.mark.parametrize("head_num", [1, 8])
@pytest.mark.parametrize("embed_dim", [128, 256, 512])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_multi_head_attention(batch_size, seq_length, head_num, embed_dim, dtype):
    x = torch.randn(batch_size, seq_length, embed_dim, dtype=dtype, device='cuda')
    multi_head_attention_module = SimpleMultiHeadAttention(head_num, embed_dim, dtype)
    mask = torch.triu(torch.ones(seq_length, seq_length, dtype=dtype, device='cuda')) * -1e9 # (seq_len, seq_len)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    output = multi_head_attention_module(x, mask)
    assert output.shape == (batch_size, seq_length, embed_dim)
