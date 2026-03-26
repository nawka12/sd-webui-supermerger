# scripts/mergers/methods.py
"""
Dispatch layer: maps supermerger calcmodes to sd-mecha math functions.

Calling conventions:
  Category A (StateDict-typed): weighted_sum, add_difference
    -> use _call(fn, key, *tensors, **kwargs) wrapper
  Category B (bare Tensor): slerp, cosine, ties, dropout primitives
    -> call fn.__wrapped__() directly
  Category C (recipe builders): wrappers.dropout, wrappers.add_ties_with_dare
    -> NOT called directly; their chains are replicated here using A/B primitives
"""
import os
import sys
import torch
from torch import Tensor

# -- Add sd_mecha submodule to path --
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_here, '..', 'sd_mecha'))

from sd_mecha.extensions.builtin.merge_methods import linear as _mm_linear
from sd_mecha.extensions.builtin.merge_methods import ties as _mm_ties
from sd_mecha.extensions.builtin.merge_methods import cosine as _mm_cosine

# Inspect add_cosine_a to confirm Category (bare Tensor vs StateDict).
# If it needs StateDict, replace direct calls below with _call().
import inspect as _inspect
_cosine_a_src = _inspect.getsource(_mm_cosine.add_cosine_a.__wrapped__)
_COSINE_IS_STATEDICT = "kwargs[" in _cosine_a_src or "[key]" in _cosine_a_src

IN_SCOPE_CALCMODES: frozenset = frozenset({
    "normal",
    "cosineA",
    "cosineB",
    "slerp",
    "ties_sum",
    "add_ties_with_dare",
    "dropout",
})

# Modes that only support 2-model merging (Weight or Add)
TWO_MODEL_ONLY: frozenset = frozenset({
    "slerp", "ties_sum", "add_ties_with_dare", "dropout",
})


def _call(fn, key: str, *tensors: Tensor, **kwargs) -> Tensor:
    """
    Call a Category A (StateDict-typed) sd-mecha __wrapped__ function on
    individual tensors by wrapping each as a single-key dict.
    Passes key= into kwargs as required by the function body.
    The __wrapped__ function returns a Tensor directly (not a dict).
    """
    wrapped = [{key: t} for t in tensors]
    return fn.__wrapped__(*wrapped, key=key, **kwargs)


def _cosine_call(fn, key: str, t0: Tensor, t1: Tensor, alpha: Tensor) -> Tensor:
    """Dispatch cosine functions handling either Category A or B."""
    if _COSINE_IS_STATEDICT:
        return _call(fn, key, t0, t1, alpha=alpha)
    return fn.__wrapped__(t0, t1, alpha=alpha)


def dispatch(calcmode: str, mode: str, key: str,
             t0: Tensor, t1: Tensor, t2, alpha: float, beta: float) -> Tensor:
    """
    Apply the sd-mecha merge function for `calcmode` to a single tensor pair.

    Args:
        calcmode: one of IN_SCOPE_CALCMODES
        mode:     supermerger mode string ("Weight", "Add", "Triple", "Twice")
        key:      state dict key (e.g. "model.diffusion_model.input_blocks.0.0.weight")
        t0:       base model tensor (Model A)
        t1:       secondary tensor (Model B, or pre-subtracted B-C for Add mode)
        t2:       tertiary tensor (Model C) or None
        alpha:    resolved per-block alpha (Python float)
        beta:     resolved per-block beta (Python float)

    Returns:
        Merged tensor with same dtype as t0.
    """
    if calcmode not in IN_SCOPE_CALCMODES:
        raise ValueError(f"Unknown calcmode for methods.dispatch: {calcmode!r}")

    a = torch.tensor(alpha, dtype=torch.float32)
    b = torch.tensor(beta, dtype=torch.float32)

    # Mode string convention: dispatch() receives the full UI mode string, e.g.
    # "Weight sum", "Add difference", "Triple sum", "sum Twice".
    # We use substring matching to identify the mode, consistent with how
    # mergers.py uses MODES = ["Weight", "Add", "Triple", "Twice"].
    # No import from mergers.py -- avoids circular imports.
    _MODE_ADD    = "Add"
    _MODE_TRIPLE = "Triple"
    _MODE_TWICE  = "Twice"

    # ---- normal mode (Weight / Add / Triple / Twice) ----
    if calcmode == "normal":
        if _MODE_ADD in mode:
            return _call(_mm_linear.add_difference, key, t0, t1, alpha=a)

        if _MODE_TRIPLE in mode:  # Triple: A*(1-alpha-beta) + B*alpha + C*beta
            if t2 is None:
                raise ValueError(f"calcmode 'normal' in mode '{mode}' requires t2 (model C)")
            if alpha + beta != 0:
                ratio = torch.tensor(beta / (alpha + beta), dtype=torch.float32)
                mid = _call(_mm_linear.weighted_sum, key, t1, t2, alpha=ratio)
                return _call(_mm_linear.weighted_sum, key, t0, mid,
                             alpha=torch.tensor(alpha + beta, dtype=torch.float32))
            return t0

        if _MODE_TWICE in mode:  # Twice: (A->B) then (result->C)
            if t2 is None:
                raise ValueError(f"calcmode 'normal' in mode '{mode}' requires t2 (model C)")
            mid = _call(_mm_linear.weighted_sum, key, t0, t1, alpha=a)
            return _call(_mm_linear.weighted_sum, key, mid, t2, alpha=b)

        # Weight mode (fallthrough)
        return _call(_mm_linear.weighted_sum, key, t0, t1, alpha=a)

    # ---- cosine modes ----
    if calcmode == "cosineA":
        return _cosine_call(_mm_cosine.add_cosine_a, key, t0, t1, a)

    if calcmode == "cosineB":
        return _cosine_call(_mm_cosine.add_cosine_b, key, t0, t1, a)

    # ---- slerp ----
    if calcmode == "slerp":
        return _mm_linear.slerp.__wrapped__(t0, t1, alpha=a)

    # ---- ties_sum: operates on deltas ----
    if calcmode == "ties_sum":
        delta = t1.float() - t0.float()
        trimmed = _mm_ties.ties_sum.__wrapped__(delta, k=1.0, vote_sgn=False)
        return (t0.float() + trimmed).to(t0.dtype)

    # ---- dropout (ties.dropout primitive, not wrappers.dropout) ----
    if calcmode == "dropout":
        delta = t1.float() - t0.float()
        dared = _mm_ties.dropout.__wrapped__(
            delta, probability=0.9, rescale=1.0, seed=None)
        return _call(_mm_linear.add_difference, key, t0, dared, alpha=a)

    # ---- add_ties_with_dare: 3-step chain ----
    if calcmode == "add_ties_with_dare":
        delta = t1.float() - t0.float()
        # probability must be plain float -- math.isclose() is called on it internally.
        # apply_stock=True avoids get_model_stock_t which requires >=2 deltas for pairwise
        # cosine similarity; with a single delta (one model pair) it would raise RuntimeError.
        result_delta = _mm_ties.ties_sum_with_dropout.__wrapped__(
            delta,
            probability=0.9,   # plain float
            della_eps=0.0,
            rescale=True,
            k=1.0,
            vote_sgn=False,
            apply_stock=True,
            cos_eps=1e-6,
            apply_median=False,
            eps=1e-6,
            maxiter=100,
            ftol=1e-20,
            seed=None,
        )
        return _call(_mm_linear.add_difference, key, t0, result_delta, alpha=a)

    raise ValueError(f"Unknown calcmode for methods.dispatch: {calcmode!r}")
