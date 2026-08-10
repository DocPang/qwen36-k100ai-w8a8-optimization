"""R244: direct strided Q/K/V reads for the single-spec GDN recurrent path.

R243 removed the recurrent output temporary/copy. The remaining single-request MTP3
spec path still takes the contiguous causal-conv output [T, Q|K|V], creates three
split views, then materializes Q/K/V with three `.contiguous()` D2D copies before the
recurrent kernel.

R244 keeps those split tensors as non-contiguous views and uses a stride-aware clone
of the *same* upstream Triton recurrent kernel. The arithmetic/reduction/state update
is unchanged; only Q/K/V pointer arithmetic uses explicit token/head strides.

The fast path is restricted to the same one-request all-spec <=4-token condition as
R243. Prefill, non-spec and mixed spec/non-spec paths are untouched.
"""
from __future__ import annotations

import inspect
import textwrap
import torch
from vllm.triton_utils import triton

from r244_strided_gdn_ops import (
    fused_sigmoid_gating_delta_rule_update_kernel as _strided_kernel,
)

_installed = False


def _r244_rearrange_views(self, mixed_qkv):
    if mixed_qkv is None:
        return None, None, None
    qdim = self.key_dim // self.tp_size
    kdim = self.key_dim // self.tp_size
    vdim = self.value_dim // self.tp_size
    query, key, value = torch.split(mixed_qkv, [qdim, kdim, vdim], dim=-1)
    t = mixed_qkv.size(0)
    hk = self.num_k_heads // self.tp_size
    hv = self.num_v_heads // self.tp_size
    # These are views of the same contiguous [T,Q|K|V] causal-conv output.
    # Token stride remains Q+K+V while head/feature strides remain packed.
    query = query.view(1, t, hk, self.head_k_dim)
    key = key.view(1, t, hk, self.head_k_dim)
    value = value.view(1, t, hv, self.head_v_dim)
    return query, key, value


def _r244_fused_gdn_into_strided(
    output,
    *,
    A_log,
    a,
    b,
    dt_bias,
    q,
    k,
    v,
    initial_state,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    use_qk_l2norm_in_kernel=True,
):
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1
    assert initial_state is not None
    assert output.is_contiguous()
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1

    final_state = initial_state
    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    _strided_kernel[(NK, NV, N * HV)](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
        q=q,
        k=k,
        v=v,
        o=output,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        scale=K**-0.5,
        N=N,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=final_state.stride(0),
        stride_indices_seq=stride_indices_seq,
        stride_indices_tok=stride_indices_tok,
        stride_qt=q.stride(1),
        stride_qh=q.stride(2),
        stride_kt=k.stride(1),
        stride_kh=k.stride(2),
        stride_vt=v.stride(1),
        stride_vh=v.stride(2),
        INPLACE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=False,
        num_warps=4,
        num_stages=3,
    )
    return output.unsqueeze(0), final_state


_OLD_REARRANGE = '''    query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
    query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
        mixed_qkv_non_spec
    )
'''

_NEW_REARRANGE = '''    _k100_r244_direct_spec = (
        spec_sequence_masks is not None
        and attn_metadata.num_spec_decodes == 1
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes == 0
        and num_actual_tokens > 0
        and num_actual_tokens <= 4
    )
    if _k100_r244_direct_spec:
        query_spec, key_spec, value_spec = _k100_r244_rearrange_views(
            self, mixed_qkv_spec
        )
    else:
        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(mixed_qkv_spec)
    query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
        mixed_qkv_non_spec
    )
'''

_OLD_SPEC = '''    # 2.1: Process the multi-query part
    if spec_sequence_masks is not None:
        core_attn_out_spec, last_recurrent_state = (
            fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                inplace_final_state=True,
                cu_seqlens=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        )
    else:
        core_attn_out_spec, last_recurrent_state = None, None
'''

_NEW_SPEC = '''    # 2.1: Process the multi-query part
    if spec_sequence_masks is not None:
        if _k100_r244_direct_spec:
            core_attn_out_spec, last_recurrent_state = _k100_r244_fused_gdn_into_strided(
                core_attn_out[:num_actual_tokens],
                A_log=self.A_log,
                a=a,
                b=b,
                dt_bias=self.dt_bias,
                q=query_spec,
                k=key_spec,
                v=value_spec,
                initial_state=ssm_state,
                cu_seqlens=spec_query_start_loc[
                    : attn_metadata.num_spec_decodes + 1
                ],
                ssm_state_indices=spec_state_indices_tensor,
                num_accepted_tokens=num_accepted_tokens,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out_spec, last_recurrent_state = (
                fused_sigmoid_gating_delta_rule_update(
                    A_log=self.A_log,
                    a=a,
                    b=b,
                    dt_bias=self.dt_bias,
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[
                        : attn_metadata.num_spec_decodes + 1
                    ],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                )
            )
    else:
        core_attn_out_spec, last_recurrent_state = None, None
'''

_OLD_MERGE = '''    # 3. Merge core attention output
    if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
        merged_out = torch.empty(
            (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
            dtype=core_attn_out_non_spec.dtype,
            device=core_attn_out_non_spec.device,
        )
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    elif spec_sequence_masks is not None:
        core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
'''

_NEW_MERGE = '''    # 3. Merge core attention output
    if spec_sequence_masks is not None and core_attn_out_non_spec is not None:
        merged_out = torch.empty(
            (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
            dtype=core_attn_out_non_spec.dtype,
            device=core_attn_out_non_spec.device,
        )
        merged_out.index_copy_(1, spec_token_indx, core_attn_out_spec)
        merged_out.index_copy_(1, non_spec_token_indx, core_attn_out_non_spec)
        core_attn_out[:num_actual_tokens] = merged_out.squeeze(0)
    elif spec_sequence_masks is not None:
        if not _k100_r244_direct_spec:
            core_attn_out[:num_actual_tokens] = core_attn_out_spec.squeeze(0)
    else:
        core_attn_out[:num_actual_tokens] = core_attn_out_non_spec.squeeze(0)
'''


def install() -> None:
    global _installed
    if _installed:
        return
    import vllm.model_executor.models.qwen3_next as qn

    cls = qn.Qwen3NextGatedDeltaNet
    source = textwrap.dedent(inspect.getsource(cls._forward_core))
    for anchor, name in (
        (_OLD_REARRANGE, "rearrange"),
        (_OLD_SPEC, "spec recurrent"),
        (_OLD_MERGE, "merge"),
    ):
        if anchor not in source:
            raise RuntimeError(f"R244 source gate failed: {name} block not found")
    source = source.replace(_OLD_REARRANGE, _NEW_REARRANGE, 1)
    source = source.replace(_OLD_SPEC, _NEW_SPEC, 1)
    source = source.replace(_OLD_MERGE, _NEW_MERGE, 1)
    ns = dict(qn.__dict__)
    ns["_k100_r244_rearrange_views"] = _r244_rearrange_views
    ns["_k100_r244_fused_gdn_into_strided"] = _r244_fused_gdn_into_strided
    exec(compile(source, "<k100-r244-qwen-gdn-core>", "exec"), ns)
    cls._forward_core = ns["_forward_core"]
    _installed = True
    print(
        "[K100 R244 GDN strided QKV] single-spec recurrent kernel reads split views directly",
        flush=True,
    )
