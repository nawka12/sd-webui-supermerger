# scripts/mergers/lycoris_compose.py
"""
Pure-tensor LyCORIS composition helpers.
No imports from sd-mecha, no A1111/WebUI dependencies.
All compose functions move tensors to CPU before computation.
"""
import math
import torch


def _rebuild_tucker(t, wa, wb):
    """Tucker decomposition rebuild: einsum over core t with factors wa, wb.

    t:  (i, j, ...)
    wa: (i, p)
    wb: (j, r)
    -> (p, r, ...)
    """
    return torch.einsum("i j ..., i p, j r -> p r ...", t, wa, wb)


def _compose_lora(lora_sd, key):
    """Compose standard LoRA (and LoCon conv2d) tensors into a weight delta."""
    down = lora_sd[key + ".lora_down.weight"].cpu()
    up = lora_sd[key + ".lora_up.weight"].cpu()
    alpha_val = lora_sd.get(key + ".alpha")
    dim = down.shape[0]
    scale = (float(alpha_val) / dim) if alpha_val is not None else 1.0

    mid_key = key + ".lora_mid.weight"
    if mid_key in lora_sd:
        mid = lora_sd[mid_key].cpu()
        # Tucker mid: wa = up.T (rank, out), wb = down flattened (rank, in*...)
        wa = up.view(up.size(0), -1).t()
        wb = down.view(dim, -1)
        delta = _rebuild_tucker(mid, wa, wb)
    elif down.dim() == 4:
        # LoCon conv2d: compose via matrix multiply then reshape
        delta = (
            up.view(up.size(0), -1) @ down.view(dim, -1)
        ).view(up.size(0), down.size(1), *down.shape[2:])
    else:
        delta = up @ down

    return delta * scale


def _compose_lokr(lora_sd, key):
    """Compose LoKr (Kronecker product) tensors into a weight delta.
    All operations run on CPU to avoid VRAM spikes from kron.
    """
    lora_dim = None

    # Resolve w1
    if key + ".lokr_w1" in lora_sd:
        w1 = lora_sd[key + ".lokr_w1"].cpu()
    else:
        w1a = lora_sd[key + ".lokr_w1_a"].cpu()
        w1b = lora_sd[key + ".lokr_w1_b"].cpu()
        w1 = w1a @ w1b
        lora_dim = w1b.shape[0]

    # Resolve w2
    if key + ".lokr_t2" in lora_sd:
        t2 = lora_sd[key + ".lokr_t2"].cpu()
        w2a = lora_sd[key + ".lokr_w2_a"].cpu()
        w2b = lora_sd[key + ".lokr_w2_b"].cpu()
        w2 = _rebuild_tucker(t2, w2a, w2b)
        lora_dim = lora_dim if lora_dim is not None else w2b.shape[0]
    elif key + ".lokr_w2_a" in lora_sd:
        w2a = lora_sd[key + ".lokr_w2_a"].cpu()
        w2b = lora_sd[key + ".lokr_w2_b"].cpu()
        w2_b_flat = w2b.flatten(1) if w2b.dim() > 1 else w2b
        w2 = w2a @ w2_b_flat
        lora_dim = lora_dim if lora_dim is not None else w2b.shape[0]
    else:
        w2 = lora_sd[key + ".lokr_w2"].cpu()

    # Broadcast w1 dims to match w2 for kron
    while w1.dim() < w2.dim():
        w1 = w1.unsqueeze(-1)

    # alpha/dim scaling only applies when lora_dim is known (i.e. a factorized path
    # was taken). For fully pre-composed plain w1/w2 matrices there is no rank
    # dimension, so alpha scaling is not applicable.
    alpha_val = lora_sd.get(key + ".alpha")
    if alpha_val is not None and lora_dim is not None:
        alpha_f = float(alpha_val)
        scale = alpha_f / lora_dim if math.isfinite(alpha_f) else 1.0
    else:
        scale = 1.0

    return torch.kron(w1, w2) * scale


def _compose_loha(lora_sd, key):
    """Compose LoHa (Hadamard product of two low-rank factors) into a weight delta."""
    w1a = lora_sd[key + ".hada_w1_a"].cpu()
    w1b = lora_sd[key + ".hada_w1_b"].cpu()
    w2a = lora_sd[key + ".hada_w2_a"].cpu()
    w2b = lora_sd[key + ".hada_w2_b"].cpu()

    t1_key = key + ".hada_t1"
    t2_key = key + ".hada_t2"

    if t1_key in lora_sd:
        part1 = _rebuild_tucker(lora_sd[t1_key].cpu(), w1a, w1b)
    else:
        part1 = w1a @ w1b

    if t2_key in lora_sd:
        part2 = _rebuild_tucker(lora_sd[t2_key].cpu(), w2a, w2b)
    else:
        part2 = w2a @ w2b

    alpha_val = lora_sd.get(key + ".alpha")
    dim = w1b.shape[0]
    scale = (float(alpha_val) / dim) if alpha_val is not None else 1.0

    return (part1 * part2) * scale


def compose_delta(lora_sd, module_key, target_shape=None):
    """Compose LyCORIS tensors for module_key into a weight delta.

    Detects algorithm by probing key existence in order: LoHa → LoKr → LoRA.
    Applies alpha/dim scaling internally.

    Args:
        lora_sd: state dict mapping full key names to tensors.
        module_key: key prefix (e.g. "lora_unet_down_blocks_0_attentions_0_to_q").
        target_shape: if provided, reshape the delta before returning.

    Returns:
        Composed delta tensor (CPU), or None if no known algorithm keys found.
    """
    if module_key + ".hada_w1_a" in lora_sd:
        delta = _compose_loha(lora_sd, module_key)
    elif (module_key + ".lokr_w1" in lora_sd or
          module_key + ".lokr_w1_a" in lora_sd):
        delta = _compose_lokr(lora_sd, module_key)
    elif module_key + ".lora_down.weight" in lora_sd:
        delta = _compose_lora(lora_sd, module_key)
    else:
        return None

    if target_shape is not None:
        delta = delta.reshape(target_shape)

    return delta
