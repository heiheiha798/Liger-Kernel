import pytest
import torch
import torch.nn.functional as F

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

import liger_kernel.ops.multi_token_attention as multi_token_attention_ops

from liger_kernel.transformers.functional import liger_multi_token_attention
from liger_kernel.transformers.multi_token_attention import LigerMultiTokenAttention
from liger_kernel.utils import infer_device

device = infer_device()
set_seed()


def _make_mask(L, device):
    tril = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    return tril.view(1, 1, L, L)


class TorchMultiTokenAttention(torch.nn.Module):
    def __init__(self, C_in, C_out, K, groups, bias, dtype, device):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(C_out, C_in // groups, K, K, dtype=dtype, device=device))
        self.bias = torch.nn.Parameter(torch.empty(C_out, dtype=dtype, device=device)) if bias else None
        self.K = K
        self.groups = groups

    def forward(self, scores):
        B, C_in, L, _ = scores.shape
        mask = _make_mask(L, scores.device)
        inf = torch.tensor(-1e9, device=scores.device, dtype=scores.dtype)
        zero = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        s_inf = scores.masked_fill(~mask, inf)
        probs = F.softmax(s_inf, dim=-1)
        out_c = F.conv2d(probs, self.weight, self.bias, stride=1, padding=self.K // 2, groups=self.groups)
        return out_c.masked_fill(~mask, zero)


@pytest.mark.skipif(device == "xpu", reason="Skip for xpu")
@pytest.mark.parametrize(
    "B,C_in,C_out,L,K,groups",
    [
        (2, 4, 4, 8, 3, 1),
        (1, 2, 2, 5, 1, 1),
        (3, 6, 6, 6, 3, 1),
        (1, 4, 4, 33, 5, 4),
        (1, 8, 8, 128, 5, 2),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 2e-2, 2e-2),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_multi_token_attention_correctness(B, C_in, C_out, L, K, groups, bias, dtype, atol, rtol):
    set_seed(42)
    scores = torch.randn(B, C_in, L, L, device=device, dtype=dtype)  # input

    ref_attn = TorchMultiTokenAttention(
        C_in=C_in, C_out=C_out, K=K, groups=groups, bias=bias, dtype=dtype, device=device
    )

    liger_attn = (
        LigerMultiTokenAttention(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=K,
            stride=1,
            padding=K // 2,
            groups=groups,
            bias=bias,
        )
        .to(device)
        .to(dtype)
    )

    with torch.no_grad():
        ref_attn.weight.copy_(liger_attn.weight)
        if bias:
            ref_attn.bias.copy_(liger_attn.bias)

    scores1 = scores.detach().clone().requires_grad_(True)
    scores2 = scores.detach().clone().requires_grad_(True)

    out1 = liger_attn(scores1)
    out2 = ref_attn(scores2)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(out1)
    out1.backward(gradient=grad_output)
    out2.backward(gradient=grad_output.clone())

    assert_verbose_allclose(scores1.grad, scores2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_attn.weight.grad, ref_attn.weight.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(liger_attn.bias.grad, ref_attn.bias.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B,C_in,C_out,L,K,groups",
    [
        (2, 4, 4, 8, 3, 1),
        (1, 2, 2, 5, 1, 1),
        (3, 6, 6, 6, 3, 1),
        (1, 4, 4, 33, 5, 4),
        (1, 8, 8, 64, 3, 2),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 2e-2, 2e-2),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_multi_token_attention_functional(B, C_in, C_out, L, K, groups, bias, dtype, atol, rtol):
    scores = torch.randn(B, C_in, L, L, device=device, dtype=dtype)

    ref_attn = TorchMultiTokenAttention(
        C_in=C_in, C_out=C_out, K=K, groups=groups, bias=bias, dtype=dtype, device=device
    )

    weight = torch.empty(C_out, C_in // groups, K, K, device=device, dtype=dtype)
    torch.nn.init.kaiming_uniform_(weight, a=5**0.5)
    if bias:
        bias_tensor = torch.empty(C_out, device=device, dtype=dtype)
        torch.nn.init.zeros_(bias_tensor)
    else:
        bias_tensor = None

    with torch.no_grad():
        ref_attn.weight.copy_(weight)
        if bias:
            ref_attn.bias.copy_(bias_tensor)

    scores1 = scores.detach().clone().requires_grad_(True)
    scores2 = scores.detach().clone().requires_grad_(True)
    weight1 = weight.detach().clone().requires_grad_(True)
    if bias:
        bias1 = bias_tensor.detach().clone().requires_grad_(True)
    else:
        bias1 = None

    out1 = liger_multi_token_attention(scores1, weight1, bias1, stride=1, padding=K // 2, groups=groups)
    out2 = ref_attn(scores2)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(out1)
    out1.backward(gradient=grad_output)
    out2.backward(gradient=grad_output.clone())

    assert_verbose_allclose(scores1.grad, scores2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, ref_attn.weight.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, ref_attn.bias.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 2e-2, 2e-2),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            marks=pytest.mark.skipif(
                not supports_bfloat16(),
                reason="bfloat16 not supported on this device",
            ),
        ),
    ],
)
def test_multi_token_attention_noncontiguous_scores(dtype, atol, rtol):
    B, C_in, C_out, L, K, groups = 1, 4, 4, 33, 3, 2
    scores = torch.randn(B, C_in, L, L, device=device, dtype=dtype).transpose(-1, -2)
    assert not scores.is_contiguous()
    weight = torch.randn(C_out, C_in // groups, K, K, device=device, dtype=dtype)
    bias = torch.randn(C_out, device=device, dtype=dtype)
    grad_output = torch.randn(B, C_out, L, L, device=device, dtype=dtype)

    scores1 = scores.detach().requires_grad_(True)
    scores2 = scores.detach().requires_grad_(True)
    weight1 = weight.detach().clone().requires_grad_(True)
    weight2 = weight.detach().clone().requires_grad_(True)
    bias1 = bias.detach().clone().requires_grad_(True)
    bias2 = bias.detach().clone().requires_grad_(True)

    out1 = liger_multi_token_attention(scores1, weight1, bias1, padding=K // 2, groups=groups)
    mask = _make_mask(L, scores2.device)
    inf = torch.tensor(-1e9, device=scores2.device, dtype=scores2.dtype)
    probs = F.softmax(scores2.masked_fill(~mask, inf), dim=-1)
    out2 = F.conv2d(probs, weight2, bias2, padding=K // 2, groups=groups).masked_fill(~mask, 0.0)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)
    out1.backward(gradient=grad_output)
    out2.backward(gradient=grad_output.clone())
    assert_verbose_allclose(scores1.grad, scores2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)


def test_multi_token_attention_reuses_softmax_launch_metadata(monkeypatch):
    metadata = {}
    original_forward = multi_token_attention_ops._softmax_forward
    original_backward = multi_token_attention_ops._softmax_backward

    def capture_forward(scores):
        result = original_forward(scores)
        metadata["forward"] = result[1:]
        return result

    def capture_backward(grad_output, output, BLOCK_SIZE, num_warps, multi_block_launch):
        metadata["backward"] = (BLOCK_SIZE, num_warps, multi_block_launch)
        return original_backward(grad_output, output, BLOCK_SIZE, num_warps, multi_block_launch)

    monkeypatch.setattr(multi_token_attention_ops, "_softmax_forward", capture_forward)
    monkeypatch.setattr(multi_token_attention_ops, "_softmax_backward", capture_backward)

    scores = torch.randn(1, 2, 33, 33, device=device, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(2, 2, 3, 3, device=device, dtype=torch.float32, requires_grad=True)
    output = liger_multi_token_attention(scores, weight, padding=1)
    output.backward(torch.randn_like(output))

    assert metadata["backward"] == metadata["forward"]


class TorchSparseMultiTokenAttention(TorchMultiTokenAttention):
    def forward(self, scores):
        B, C_in, L, _ = scores.shape
        mask = _make_mask(L, scores.device)
        inf = torch.tensor(-1e9, device=scores.device, dtype=scores.dtype)
        zero = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        s_inf = scores.masked_fill(~mask, inf)
        dim = -1
        z = s_inf
        z_sorted, _ = torch.sort(z, dim=dim, descending=True)
        cum_sum = torch.cumsum(z_sorted, dim=dim)
        k_indices = torch.arange(1, L + 1, device=z.device, dtype=z.dtype).view(1, 1, 1, L)
        is_positive = z_sorted > -1e8
        condition = (1 + k_indices * z_sorted > cum_sum) & is_positive
        k_sparsemax = torch.sum(condition, dim=dim, keepdim=True)
        k_sparsemax_safe = torch.max(k_sparsemax, torch.ones_like(k_sparsemax))
        cum_sum_k = torch.gather(cum_sum, dim=dim, index=k_sparsemax_safe.long() - 1)
        tau = (cum_sum_k - 1) / k_sparsemax_safe.to(z.dtype)
        tau = torch.where(k_sparsemax == 0, torch.full_like(tau, float("inf")), tau)
        probs = torch.clamp(z - tau, min=0)
        out_c = F.conv2d(probs, self.weight, self.bias, stride=1, padding=self.K // 2, groups=self.groups)
        return out_c.masked_fill(~mask, zero)


# NOTE(tcc): Unknown failure on xpu. Issue #761
@pytest.mark.skipif(device == "xpu", reason="Skip for xpu")
@pytest.mark.parametrize(
    "B,C_in,C_out,L,K,groups",
    [
        (2, 4, 4, 8, 3, 1),
        (1, 2, 2, 5, 1, 1),
        (3, 6, 6, 6, 3, 1),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-4, 5e-4),
    ],
)
def test_sparse_multi_token_attention_correctness(B, C_in, C_out, L, K, groups, bias, dtype, atol, rtol):
    set_seed()
    scores = torch.randn(B, C_in, L, L, device=device, dtype=dtype)

    ref_attn = TorchSparseMultiTokenAttention(
        C_in=C_in, C_out=C_out, K=K, groups=groups, bias=bias, dtype=dtype, device=device
    )

    liger_attn = (
        LigerMultiTokenAttention(
            in_channels=C_in,
            out_channels=C_out,
            kernel_size=K,
            stride=1,
            padding=K // 2,
            groups=groups,
            bias=bias,
            sparse=True,
        )
        .to(device)
        .to(dtype)
    )

    torch.nn.init.kaiming_uniform_(liger_attn.weight, a=5**0.5)
    if bias:
        torch.nn.init.zeros_(liger_attn.bias)

    with torch.no_grad():
        ref_attn.weight.copy_(liger_attn.weight)
        if bias:
            ref_attn.bias.copy_(liger_attn.bias)

    scores1 = scores.detach().clone().requires_grad_(True)
    scores2 = scores.detach().clone().requires_grad_(True)

    out1 = liger_attn(scores1)
    out2 = ref_attn(scores2)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(out1)
    out1.backward(gradient=grad_output)
    out2.backward(gradient=grad_output.clone())

    assert_verbose_allclose(scores1.grad, scores2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(liger_attn.weight.grad, ref_attn.weight.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(liger_attn.bias.grad, ref_attn.bias.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "B,C_in,C_out,L,K,groups",
    [
        (2, 4, 4, 8, 3, 1),
        (1, 2, 2, 5, 1, 1),
        (3, 6, 6, 6, 3, 1),
    ],
)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 5e-4, 5e-4),
    ],
)
def test_sparse_multi_token_attention_functional(B, C_in, C_out, L, K, groups, bias, dtype, atol, rtol):
    set_seed()
    scores = torch.randn(B, C_in, L, L, device=device, dtype=dtype)

    ref_attn = TorchSparseMultiTokenAttention(
        C_in=C_in, C_out=C_out, K=K, groups=groups, bias=bias, dtype=dtype, device=device
    )

    weight = torch.empty(C_out, C_in // groups, K, K, device=device, dtype=dtype)
    torch.nn.init.kaiming_uniform_(weight, a=5**0.5)
    if bias:
        bias_tensor = torch.empty(C_out, device=device, dtype=dtype)
        torch.nn.init.zeros_(bias_tensor)
    else:
        bias_tensor = None

    with torch.no_grad():
        ref_attn.weight.copy_(weight)
        if bias:
            ref_attn.bias.copy_(bias_tensor)

    scores1 = scores.detach().clone().requires_grad_(True)
    scores2 = scores.detach().clone().requires_grad_(True)
    weight1 = weight.detach().clone().requires_grad_(True)
    if bias:
        bias1 = bias_tensor.detach().clone().requires_grad_(True)
    else:
        bias1 = None

    out1 = liger_multi_token_attention(scores1, weight1, bias1, stride=1, padding=K // 2, groups=groups, sparse=True)
    out2 = ref_attn(scores2)

    assert_verbose_allclose(out1, out2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(out1)
    out1.backward(gradient=grad_output)
    out2.backward(gradient=grad_output.clone())

    assert_verbose_allclose(scores1.grad, scores2.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(weight1.grad, ref_attn.weight.grad, atol=atol, rtol=rtol)
    if bias:
        assert_verbose_allclose(bias1.grad, ref_attn.bias.grad, atol=atol, rtol=rtol)
