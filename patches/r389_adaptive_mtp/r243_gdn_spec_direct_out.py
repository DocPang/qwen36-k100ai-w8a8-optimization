"""R243: write single-request speculative GDN recurrent output directly to core_attn_out.

Qwen3Next's speculative GDN path currently allocates a temporary output in
fused_sigmoid_gating_delta_rule_update() and then copies it into the custom-op output
buffer.  For the one-request, all-spec MTP3 hot path (R239/R241 metadata), invoke the
same upstream Triton kernel with core_attn_out as its output pointer and skip the D2D
copy.  Kernel math/config/state updates remain identical.
"""
from __future__ import annotations

import inspect
import textwrap
from vllm.triton_utils import triton

_installed = False


def _r243_fused_gdn_into(
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
    from vllm.model_executor.layers.fla.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update_kernel as _kernel,
    )

    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1
    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 32)
    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    assert NK == 1
    assert initial_state is not None
    assert output.is_contiguous()

    final_state = initial_state
    if ssm_state_indices is None:
        stride_indices_seq, stride_indices_tok = 1, 1
    elif ssm_state_indices.ndim == 1:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride(0), 1
    else:
        stride_indices_seq, stride_indices_tok = ssm_state_indices.stride()

    _kernel[(NK, NV, N * HV)](
        A_log=A_log,
        a=a.contiguous(),
        b=b.contiguous(),
        dt_bias=dt_bias,
        beta=1.0,
        threshold=20.0,
        q=q.contiguous(),
        k=k.contiguous(),
        v=v.contiguous(),
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
        INPLACE_FINAL_STATE=True,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=False,
        num_warps=4,
        num_stages=3,
    )
    # Match the stock wrapper return shape after squeezing the NK dimension.
    return output.unsqueeze(0), final_state


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
    _k100_r243_direct_spec = (
        spec_sequence_masks is not None
        and attn_metadata.num_spec_decodes == 1
        and attn_metadata.num_prefills == 0
        and attn_metadata.num_decodes == 0
        and num_actual_tokens > 0
        and num_actual_tokens <= 4
    )
    if spec_sequence_masks is not None:
        if _k100_r243_direct_spec:
            core_attn_out_spec, last_recurrent_state = _k100_r243_fused_gdn_into(
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
        if not _k100_r243_direct_spec:
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
    if _OLD_SPEC not in source or _OLD_MERGE not in source:
        raise RuntimeError("R243 source gate failed: expected Qwen GDN spec blocks not found")
    source = source.replace(_OLD_SPEC, _NEW_SPEC, 1)
    source = source.replace(_OLD_MERGE, _NEW_MERGE, 1)
    ns = dict(qn.__dict__)
    ns["_k100_r243_fused_gdn_into"] = _r243_fused_gdn_into
    exec(compile(source, "<k100-r243-qwen-gdn-core>", "exec"), ns)
    cls._forward_core = ns["_forward_core"]
    _installed = True
    print(
        "[K100 R243 GDN direct out] single-spec recurrent kernel writes core_attn_out directly",
        flush=True,
    )
