"""R245: keep Qwen3.5 BA chunk views strided on the single-spec GDN fast path.

R184's Qwen3.5 fused QKVZ+BA forward already emits BA as [all b | all a].
`ba.chunk(2)` therefore creates exact [T,32] views with stride (64,1), but the
forward immediately materializes both with `.contiguous()`.  R245 removes those
two copies from the one-request all-spec <=4-token path and teaches the R244
recurrent kernel to read the strided BA views directly.  All other GDN paths
materialize b/a inside the opaque core before executing upstream logic.
"""
from __future__ import annotations

import inspect
import textwrap
import torch
from vllm.triton_utils import triton
from r245_strided_gdn_ops import fused_sigmoid_gating_delta_rule_update_kernel as _kernel

_installed=False


def _r245_fused_gdn_into_strided(output, *, A_log, a, b, dt_bias, q, k, v,
                                 initial_state, cu_seqlens, ssm_state_indices,
                                 num_accepted_tokens, use_qk_l2norm_in_kernel=True):
    B,T,H,K,V=*k.shape,v.shape[-1]
    HV=v.shape[2]
    N=B if cu_seqlens is None else len(cu_seqlens)-1
    BK=triton.next_power_of_2(K); BV=min(triton.next_power_of_2(V),32)
    NK,NV=triton.cdiv(K,BK),triton.cdiv(V,BV)
    assert NK==1 and initial_state is not None and output.is_contiguous()
    final_state=initial_state
    if ssm_state_indices is None:
        sis,sit=1,1
    elif ssm_state_indices.ndim==1:
        sis,sit=ssm_state_indices.stride(0),1
    else:
        sis,sit=ssm_state_indices.stride()
    ratio=HV//H
    # Qwen3.5 BA chunks are 2-D [T,HV] views with row stride 64.  The helper
    # also accepts contiguous [T,HV] tensors for safety/testing.
    sat,sah,sal=a.stride(0),a.stride(1)*ratio,a.stride(1)
    sbt,sbh,sbl=b.stride(0),b.stride(1)*ratio,b.stride(1)
    _kernel[(NK,NV,N*HV)](
        A_log=A_log,a=a,b=b,dt_bias=dt_bias,beta=1.0,threshold=20.0,
        q=q,k=k,v=v,o=output,h0=initial_state,ht=final_state,
        cu_seqlens=cu_seqlens,ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,scale=K**-0.5,N=N,T=T,B=B,H=H,HV=HV,K=K,V=V,BK=BK,BV=BV,
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=final_state.stride(0),
        stride_indices_seq=sis,stride_indices_tok=sit,
        stride_qt=q.stride(1),stride_qh=q.stride(2),
        stride_kt=k.stride(1),stride_kh=k.stride(2),
        stride_vt=v.stride(1),stride_vh=v.stride(2),
        stride_at=sat,stride_ah=sah,stride_al=sal,
        stride_bt=sbt,stride_bh=sbh,stride_bl=sbl,
        INPLACE_FINAL_STATE=True,USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=False,num_warps=4,num_stages=3)
    return output.unsqueeze(0),final_state


def install():
    global _installed
    if _installed:return
    from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
    from vllm.model_executor.models.qwen3_next import Qwen3NextGatedDeltaNet
    from vllm.forward_context import get_forward_context

    # Patch the already-installed R184 Qwen3.5 forward only at the two BA
    # materializations.  Its source lives in sitecustomize.py and is inspectable.
    orig_forward=Qwen3_5GatedDeltaNet.forward
    src=textwrap.dedent(inspect.getsource(orig_forward))
    old='''    b, a = ba.chunk(2, dim=-1)\n    b = b.contiguous()\n    a = a.contiguous()\n'''
    new='''    b, a = ba.chunk(2, dim=-1)\n'''
    if old not in src:
        raise RuntimeError('R245 forward source gate failed: BA contiguous block not found')
    ns=dict(orig_forward.__globals__)
    exec(compile(src.replace(old,new,1),'<k100-r245-qwen35-forward>','exec'),ns)
    Qwen3_5GatedDeltaNet.forward=ns[orig_forward.__name__]

    orig_core=Qwen3NextGatedDeltaNet._forward_core
    if '_k100_r244_fused_gdn_into_strided' not in orig_core.__globals__:
        raise RuntimeError('R245 core source gate failed: R244 helper not installed')
    orig_core.__globals__['_k100_r244_fused_gdn_into_strided']=_r245_fused_gdn_into_strided

    def core_r245(self,mixed_qkv,b,a,core_attn_out):
        fc=get_forward_context(); md=fc.attn_metadata
        md=md[self.prefix] if isinstance(md,dict) else md
        direct=(getattr(md,'spec_sequence_masks',None) is not None
                and getattr(md,'num_spec_decodes',0)==1
                and getattr(md,'num_prefills',0)==0
                and getattr(md,'num_decodes',0)==0
                and 0 < getattr(md,'num_actual_tokens',0) <= 4)
        if not direct:
            b=b.contiguous(); a=a.contiguous()
        return orig_core(self,mixed_qkv,b,a,core_attn_out)
    Qwen3NextGatedDeltaNet._forward_core=core_r245
    _installed=True
    print('[K100 R245 GDN strided BA] single-spec reads Qwen3.5 BA chunk views directly',flush=True)
