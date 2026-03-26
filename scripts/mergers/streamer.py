# scripts/mergers/streamer.py
"""
Save path: builds the merged output dict key-by-key (releasing each
merged tensor immediately), then writes once via safetensors save_file().
The input model dicts (theta_0, theta_1, theta_2) must already be loaded.
Memory saving: the merged *result* dict is never fully resident; only one
result tensor is held at a time before being stored in output_tensors.

Called by smerge() after the preamble (architecture detection, dtype
normalization, Stage 0 pre-subtraction) when a save path is present.
"""
import numpy as np
import torch
from safetensors.torch import save_file as _sf_save_file

from mergers import methods as _methods


def merge_and_save(
    save_path: str,
    theta_0: dict,
    theta_1: dict,
    theta_2,
    calcmode: str,
    mode: str,
    base_alpha: float,
    base_beta: float,
    weights_a: list,
    weights_b: list,
    isxl: bool,
    isflux: bool,
    useblocks: bool,
    usebeta: bool,
    deep: list,
    randomer,
    lucks: dict,
    deepprint: bool,
    esettings: list,
    inex: str,
    ex_blocks: list,
    ex_elems: list,
):
    """
    Merge theta_0 with theta_1 (and optionally theta_2) key-by-key and
    stream results to save_path as a safetensors file.

    theta_0, theta_1, theta_2 must already be fully loaded and through
    the smerge() preamble (Stage 0 pre-subtraction etc.).
    """
    # Lazy import to avoid circular dependency (mergers.mergers imports streamer at module level)
    from mergers.mergers import resolve_alpha, CHCKPOINT_DICT_SKIP_ON_MERGE

    if randomer is None:
        randomer = np.zeros(3000)

    weights_a_excluded = list(weights_a) if weights_a else [None] * 110
    weights_b_excluded = list(weights_b) if weights_b else [None] * 110

    output_tensors = {}

    # Stage 1/2 — merge keys present in theta_0
    for num, key in enumerate(theta_0.keys()):
        if isflux:
            if key not in theta_1:
                continue
        else:
            if not ("model" in key and key in theta_1):
                continue
        if not ("weight" in key or "bias" in key):
            continue
        if theta_2 is not None and key not in theta_2:
            continue

        current_alpha, current_beta, skip = resolve_alpha(
            key, isxl, isflux, useblocks, usebeta,
            base_alpha, base_beta, weights_a, weights_b,
            deep, randomer, num, lucks, deepprint, esettings,
            inex, ex_blocks, ex_elems,
            weights_a_excluded, weights_b_excluded,
        )

        if skip:
            output_tensors[key] = theta_0[key]
            continue

        a_shape = list(theta_0[key].shape)
        b_shape = list(theta_1[key].shape)
        if a_shape != b_shape and a_shape[0:1] + a_shape[2:] == b_shape[0:1] + b_shape[2:]:
            t0 = theta_0[key][:, 0:4, :, :]
            slice_only = True
        else:
            t0 = theta_0[key]
            slice_only = False

        result = _methods.dispatch(
            calcmode, mode, key,
            t0, theta_1[key],
            theta_2[key] if theta_2 is not None else None,
            current_alpha, current_beta,
        )

        if slice_only:
            out_tensor = theta_0[key].clone()
            out_tensor[:, 0:4, :, :] = result
            output_tensors[key] = out_tensor
        else:
            output_tensors[key] = result

    # Stage 2/2 — copy keys from theta_1 absent in theta_0 (text encoder etc.)
    for key in theta_1.keys():
        if key in CHCKPOINT_DICT_SKIP_ON_MERGE or isflux:
            continue
        if "model" in key and key not in output_tensors:
            output_tensors[key] = theta_1[key]

    _sf_save_file(output_tensors, save_path)
