"""R241: fuse tiny GDN align-mode block-table index construction + gather.

Upstream mamba_get_block_table_tensor() executes several tiny GPU ops for align mode:
sub/floor-div/clamp, arange, broadcast add, int32->int64 cast and gather.  Qwen3.6
MTP3 TP1 repeatedly executes this metadata path with <=4 rows, block_size=1072 and
four output block ids.  Replace that sequence with one integer-only Triton kernel and
reuse a per-builder output buffer.

All other shapes/modes/configurations fall back to upstream exactly.
"""
from __future__ import annotations

import inspect
import textwrap
import torch
from vllm.triton_utils import triton, tl

_installed = False


@triton.jit
def _r241_block_table_kernel(
    block_ptr,
    seq_ptr,
    out_ptr,
    stride_b0: tl.constexpr,
    stride_b1: tl.constexpr,
    stride_o0: tl.constexpr,
    stride_o1: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NCOLS: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, NCOLS)
    seq = tl.load(seq_ptr + row).to(tl.int32)
    start = tl.maximum((seq - 1) // BLOCK_SIZE, 0)
    vals = tl.load(block_ptr + row * stride_b0 + (start + offs) * stride_b1)
    tl.store(out_ptr + row * stride_o0 + offs * stride_o1, vals)


def _r241_get_block_table(builder, block_table, seq_lens, kv_cache_spec, mamba_cache_mode):
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor as _upstream

    try:
        nrows = int(seq_lens.numel())
        block_size = int(kv_cache_spec.block_size)
        ncols = 1 + int(kv_cache_spec.num_speculative_blocks)
        if (
            mamba_cache_mode == "align"
            and 0 < nrows <= 4
            and block_size == 1072
            and ncols == 4
            and block_table.is_cuda
            and seq_lens.is_cuda
            and block_table.dtype == torch.int32
            and seq_lens.dtype == torch.int32
            and block_table.ndim == 2
        ):
            out = getattr(builder, "_k100_r241_block_table_buf", None)
            if (
                out is None
                or out.device != block_table.device
                or out.dtype != block_table.dtype
                or tuple(out.shape) != (nrows, ncols)
            ):
                out = torch.empty(
                    (nrows, ncols), dtype=block_table.dtype, device=block_table.device
                )
                builder._k100_r241_block_table_buf = out
            _r241_block_table_kernel[(nrows,)](
                block_table,
                seq_lens,
                out,
                block_table.stride(0),
                block_table.stride(1),
                out.stride(0),
                out.stride(1),
                BLOCK_SIZE=1072,
                NCOLS=4,
                num_warps=1,
            )
            return out
    except Exception:
        pass
    return _upstream(block_table, seq_lens, kv_cache_spec, mamba_cache_mode)


_OLD = '''    block_table_tensor = mamba_get_block_table_tensor(
        m.block_table_tensor,
        m.seq_lens,
        self.kv_cache_spec,
        self.vllm_config.cache_config.mamba_cache_mode,
    )
'''
_NEW = '''    block_table_tensor = _k100_r241_get_block_table(
        self,
        m.block_table_tensor,
        m.seq_lens,
        self.kv_cache_spec,
        self.vllm_config.cache_config.mamba_cache_mode,
    )
'''


def install() -> None:
    global _installed
    if _installed:
        return

    import vllm.v1.attention.backends.gdn_attn as gdn
    # R241 supersedes R239 in this stack and applies both source transformations
    # in one pass over the pristine upstream function. Dynamic functions produced
    # by R239 are intentionally not patched a second time.
    from r239_gdn_full_single_spec import (
        _OLD_MASK as _R239_OLD_MASK,
        _NEW_MASK as _R239_NEW_MASK,
        _OLD_ELSE as _R239_OLD_ELSE,
        _NEW_ELSE as _R239_NEW_ELSE,
    )

    cls = gdn.GDNAttentionMetadataBuilder
    source = textwrap.dedent(inspect.getsource(cls.build))
    if _R239_OLD_MASK not in source or _R239_OLD_ELSE not in source:
        raise RuntimeError("R241 source gate failed: pristine R239 GDN anchors not found")
    source = source.replace(_R239_OLD_MASK, _R239_NEW_MASK, 1)
    source = source.replace(_R239_OLD_ELSE, _R239_NEW_ELSE, 1)
    if _OLD not in source:
        raise RuntimeError("R241 source gate failed: GDN block-table call not found")
    source = source.replace(_OLD, _NEW, 1)
    ns = dict(gdn.__dict__)
    ns["_k100_r241_get_block_table"] = _r241_get_block_table
    exec(compile(source, "<k100-r241-gdn-build>", "exec"), ns)
    cls.build = ns["build"]
    _installed = True
    print(
        "[K100 R241 GDN combined] R239 single-spec metadata + fused tiny block-table gather installed",
        flush=True,
    )
