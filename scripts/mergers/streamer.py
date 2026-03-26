# scripts/mergers/streamer.py
"""
True streaming merge: open model files lazily with safe_open, read one tensor
per key, write result to save_path via safetensors save_file().

Peak RAM ≈ 2× model size (output_tensors accumulates all merged tensors in
memory before the final save_file() write, plus working tensors per key).

Called by smerge() when a save path is present, instead of the old dict-based
merge path.
"""
import numpy as np
import torch
from tqdm import tqdm
from safetensors.torch import save_file as _sf_save_file, safe_open as _sf_open
from contextlib import ExitStack as _ExitStack

from scripts.mergers import methods as _methods

# Mirror of mergers.PREFIXFIX/PREFIX_M — can't import from mergers at module level (circular import)
_PREFIXFIX = ("double_blocks", "single_blocks", "time_in", "vector_in", "txt_in")
_PREFIX_M = "model.diffusion_model."


def detect_arch(path: str):
    """
    Read only the safetensors header; return (isxl, isflux, keys, dtype).

    keys — list of raw file keys (before prefixer transformation).
    dtype — torch dtype of the first tensor in the file.
    Loads a single tensor only for dtype detection; no full model load.
    """
    with _sf_open(path, framework="pt", device="cpu") as sf:
        raw_keys = list(sf.keys())
        isxl = (
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight"
            in raw_keys
        )
        isflux = any("double_block" in k for k in raw_keys)
        dtype = None
        for k in raw_keys:
            try:
                dtype = sf.get_tensor(k).dtype
                break
            except Exception:
                continue
    return isxl, isflux, raw_keys, dtype


def merge_and_save(
    path_a: str,
    path_b: str,
    path_c,           # str or None
    save_path: str,
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
    add_mode: bool = False,
    device: str = "cpu",
):
    """
    True streaming merge: open model files lazily with safe_open, read
    one tensor per key, write result to save_path. Peak RAM ≈ 2× model size.
    Output tensors accumulate in a dict before the final `save_file()` write.

    path_a   — safetensors file for Model A (base)
    path_b   — safetensors file for Model B
    path_c   — safetensors file for Model C (None for two-model modes)
    add_mode — if True, compute delta = B - C per key inline (Add difference)
    """
    from scripts.mergers.mergers import resolve_alpha, CHCKPOINT_DICT_SKIP_ON_MERGE

    if randomer is None:
        randomer = np.zeros(3000)

    weights_a_excluded = list(weights_a) if weights_a else [None] * 110
    weights_b_excluded = list(weights_b) if weights_b else [None] * 110

    def _apply_prefix(raw_key: str) -> str:
        if raw_key.startswith(_PREFIXFIX):
            return _PREFIX_M + raw_key
        return raw_key

    output_tensors = {}

    with _ExitStack() as stack:
        sf_a = stack.enter_context(_sf_open(path_a, framework="pt", device=device))
        sf_b = stack.enter_context(_sf_open(path_b, framework="pt", device=device))
        sf_c = (
            stack.enter_context(_sf_open(path_c, framework="pt", device=device))
            if path_c else None
        )

        # Build internal-key → raw-file-key maps for B and C
        keys_b_map = {_apply_prefix(k): k for k in sf_b.keys()}
        keys_c_map = {_apply_prefix(k): k for k in sf_c.keys()} if sf_c else {}

        # Stage 1/2 — iterate A, merge with B (and optionally C)
        raw_keys_a = list(sf_a.keys())
        for num, raw_key_a in enumerate(tqdm(raw_keys_a, desc="Stage 1/2")):
            key = _apply_prefix(raw_key_a)  # internal key

            # Key presence filter (mirrors current streamer.py)
            if isflux:
                if key not in keys_b_map:
                    continue
            else:
                if not ("model" in key and key in keys_b_map):
                    continue
            if not ("weight" in key or "bias" in key):
                continue
            if sf_c is not None and not add_mode and key not in keys_c_map:
                continue

            current_alpha, current_beta, skip = resolve_alpha(
                key, isxl, isflux, useblocks, usebeta,
                base_alpha, base_beta, weights_a, weights_b,
                deep, randomer, num, lucks, deepprint, esettings,
                inex, ex_blocks, ex_elems,
                weights_a_excluded, weights_b_excluded,
            )

            if skip:
                output_tensors[key] = sf_a.get_tensor(raw_key_a)
                continue

            t0 = sf_a.get_tensor(raw_key_a)
            _orig_dtype = t0.dtype

            # B tensor — optionally compute Add difference inline
            raw_key_b = keys_b_map[key]
            if add_mode and sf_c is not None:
                raw_key_c = keys_c_map.get(key)
                t1_raw = sf_b.get_tensor(raw_key_b)
                if raw_key_c is not None:
                    t2_raw = sf_c.get_tensor(raw_key_c)
                    # shape mismatch check (inpaint models): compare A vs B
                    a_s, b_s = list(t0.shape), list(t1_raw.shape)
                    if a_s != b_s and a_s[0:1] + a_s[2:] == b_s[0:1] + b_s[2:]:
                        t1_slice = t1_raw[:, 0:4, :, :]
                    else:
                        t1_slice = t1_raw
                    t1 = (t1_slice.to(torch.float32) - t2_raw.to(torch.float32)).to(t1_raw.dtype)
                else:
                    t1 = torch.zeros(t1_raw.shape, dtype=_orig_dtype, device=t1_raw.device)
                t2 = None
            else:
                t1 = sf_b.get_tensor(raw_key_b)
                t2 = sf_c.get_tensor(keys_c_map[key]) if (sf_c and key in keys_c_map) else None

            # Dtype alignment: cast B and C to match A
            if t1.dtype != _orig_dtype:
                t1 = t1.to(_orig_dtype)
            if t2 is not None and t2.dtype != _orig_dtype:
                t2 = t2.to(_orig_dtype)

            # Inpaint channel-slice handling
            a_shape = list(t0.shape)
            b_shape = list(t1.shape)
            if a_shape != b_shape and a_shape[0:1] + a_shape[2:] == b_shape[0:1] + b_shape[2:]:
                t0_merge = t0[:, 0:4, :, :]
                slice_only = True
            else:
                t0_merge = t0
                slice_only = False

            result = _methods.dispatch(
                calcmode, mode, key,
                t0_merge, t1, t2,
                current_alpha, current_beta,
            )

            if slice_only:
                out_tensor = t0.clone()
                out_tensor[:, 0:4, :, :] = result.to(_orig_dtype)
                output_tensors[key] = out_tensor
            else:
                output_tensors[key] = result.to(_orig_dtype)

        # Stage 2/2 — keys in B absent in A (text encoder etc.)
        for raw_key in sf_b.keys():
            internal_key = _apply_prefix(raw_key)
            if internal_key in CHCKPOINT_DICT_SKIP_ON_MERGE or isflux:
                continue
            if "model" in internal_key and internal_key not in output_tensors:
                output_tensors[internal_key] = sf_b.get_tensor(raw_key)

    # Flux models: revert prefix so output matches file format Forge expects (bare keys)
    if isflux:
        output_tensors = {
            k[len(_PREFIX_M):] if k.startswith(_PREFIX_M) else k: v
            for k, v in output_tensors.items()
        }

    _sf_save_file(output_tensors, save_path)
