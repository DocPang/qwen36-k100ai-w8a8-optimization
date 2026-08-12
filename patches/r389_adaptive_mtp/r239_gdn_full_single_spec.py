"""R239: complete single-request speculative GDN metadata fast path.

This is a strict TP1/single-request specialization of GDNAttentionMetadataBuilder.build.
It preserves the upstream multi-request/prefill/non-spec paths verbatim, but avoids
reconstructing generic masks/index tensors when the batch contains exactly one active
speculative-decode request.

The optimization is metadata-only: no model math, routing, sampling, cache values, or
accepted-token semantics are changed.
"""
from __future__ import annotations

import inspect
import textwrap

_installed = False

_OLD_MASK = '''    spec_sequence_masks_cpu: torch.Tensor | None = None
    if (
        not self.use_spec_decode
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
        .sum()
        .item()
        == 0
    ):
        spec_sequence_masks = None
        num_spec_decodes = 0
    else:
        spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
        num_spec_decodes = spec_sequence_masks_cpu.sum().item()
        if num_spec_decodes == 0:
            spec_sequence_masks = None
            spec_sequence_masks_cpu = None
        else:
            spec_sequence_masks = spec_sequence_masks_cpu.to(
                query_start_loc.device, non_blocking=True
            )
'''

_NEW_MASK = '''    spec_sequence_masks_cpu: torch.Tensor | None = None
    _k100_r239_single_spec = False
    if (
        self.use_spec_decode
        and num_decode_draft_tokens_cpu is not None
        and num_decode_draft_tokens_cpu.numel() == 1
    ):
        _k100_r239_num_draft = int(num_decode_draft_tokens_cpu[0].item())
        if _k100_r239_num_draft > 0:
            _k100_r239_single_spec = True
            num_spec_decodes = 1
            spec_sequence_masks_cpu = getattr(self, "_k100_r239_true_cpu", None)
            if spec_sequence_masks_cpu is None:
                spec_sequence_masks_cpu = torch.ones((1,), dtype=torch.bool, device="cpu")
                self._k100_r239_true_cpu = spec_sequence_masks_cpu
            spec_sequence_masks = getattr(self, "_k100_r239_true_gpu", None)
            if spec_sequence_masks is None or spec_sequence_masks.device != query_start_loc.device:
                spec_sequence_masks = torch.ones((1,), dtype=torch.bool, device=query_start_loc.device)
                self._k100_r239_true_gpu = spec_sequence_masks
        else:
            spec_sequence_masks = None
            num_spec_decodes = 0
    elif (
        not self.use_spec_decode
        or num_decode_draft_tokens_cpu is None
        or num_decode_draft_tokens_cpu[num_decode_draft_tokens_cpu >= 0]
        .sum()
        .item()
        == 0
    ):
        spec_sequence_masks = None
        num_spec_decodes = 0
    else:
        spec_sequence_masks_cpu = num_decode_draft_tokens_cpu >= 0
        num_spec_decodes = spec_sequence_masks_cpu.sum().item()
        if num_spec_decodes == 0:
            spec_sequence_masks = None
            spec_sequence_masks_cpu = None
        else:
            spec_sequence_masks = spec_sequence_masks_cpu.to(
                query_start_loc.device, non_blocking=True
            )
'''

_OLD_ELSE = '''    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        assert spec_sequence_masks_cpu is not None
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
'''

_NEW_ELSE = '''    elif _k100_r239_single_spec:
        # For one speculative request there are no non-spec requests in this
        # scheduling step.  Avoid the generic boolean indexing / reductions /
        # argsort / advanced-indexing path and build exactly the same metadata
        # from direct views plus cached constant indices.
        num_decodes = 0
        num_prefills = 0
        num_decode_tokens = 0
        num_prefill_tokens = 0
        _k100_r239_total_query = int(query_start_loc_cpu[-1].item())
        num_spec_decode_tokens = _k100_r239_total_query
        _k100_r239_spec_token_size = min(
            self.num_spec + 1, _k100_r239_total_query
        )
        spec_token_indx = getattr(self, "_k100_r239_spec_token_indx", None)
        if (
            spec_token_indx is None
            or spec_token_indx.device != query_start_loc.device
            or spec_token_indx.numel() < self.num_spec + 1
        ):
            spec_token_indx = torch.arange(
                self.num_spec + 1,
                dtype=torch.int32,
                device=query_start_loc.device,
            )
            self._k100_r239_spec_token_indx = spec_token_indx
        spec_token_indx = spec_token_indx[:_k100_r239_spec_token_size]
        non_spec_token_indx = getattr(self, "_k100_r239_empty_i32", None)
        if non_spec_token_indx is None or non_spec_token_indx.device != query_start_loc.device:
            non_spec_token_indx = torch.empty(
                0, dtype=torch.int32, device=query_start_loc.device
            )
            self._k100_r239_empty_i32 = non_spec_token_indx
        # Boolean indexing is unnecessary for a known single True request.
        spec_state_indices_tensor = block_table_tensor[:1, : self.num_spec + 1]
        non_spec_state_indices_tensor = None
        spec_query_start_loc = query_start_loc[:2]
        non_spec_query_start_loc = None
        non_spec_query_start_loc_cpu = None
        assert num_accepted_tokens is not None
        num_accepted_tokens = num_accepted_tokens[:1]
    else:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        assert spec_sequence_masks_cpu is not None
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
'''


def install() -> None:
    global _installed
    if _installed:
        return

    import vllm.v1.attention.backends.gdn_attn as gdn

    cls = gdn.GDNAttentionMetadataBuilder
    source = textwrap.dedent(inspect.getsource(cls.build))
    if _OLD_MASK not in source:
        raise RuntimeError("R239 source gate failed: expected original GDN mask block not found")
    source = source.replace(_OLD_MASK, _NEW_MASK, 1)
    if _OLD_ELSE not in source:
        raise RuntimeError("R239 source gate failed: expected GDN spec branch head not found")
    source = source.replace(_OLD_ELSE, _NEW_ELSE, 1)

    ns = dict(gdn.__dict__)
    exec(compile(source, "<k100-r239-gdn-build>", "exec"), ns)
    cls.build = ns["build"]
    _installed = True
    print(
        "[K100 R239 GDN full single-spec] direct single-request speculative metadata path installed",
        flush=True,
    )
