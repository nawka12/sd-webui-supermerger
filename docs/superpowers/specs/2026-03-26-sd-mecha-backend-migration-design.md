# sd-mecha Backend Migration — Design Spec

**Date:** 2026-03-26
**Branch:** dev

---

## Overview

Partially migrate supermerger's merge backend to use [sd-mecha](https://github.com/ljleb/sd-mecha) for tensor math, while keeping supermerger in control of block assignment, MBW resolution, model loading, and all orchestration logic. Add four new high-value calcmodes from sd-mecha that supermerger does not currently support.

---

## Goals

- Use sd-mecha's math functions for all **in-scope** calcmodes (consistent calculations).
- Preserve the in-memory merge-and-load workflow (no UX regression).
- Provide a memory-efficient streaming path for save-to-disk merges.
- Add `slerp`, `ties_sum`, `add_ties_with_dare`, `dropout` as new calcmodes.
- Keep sd-mecha vendored as a pinned git submodule — dependencies managed via `install.py`.
- Make future sd-mecha updates easy: update submodule commit, adjust `methods.py` only.

---

## Non-Goals

- Migrating `cosineA`, `cosineB` — sd-mecha's implementations use a different algorithm (no global distribution normalization), which would silently change user results. These stay as-is.
- Migrating `trainDifference`, `extract` — complex 3-tensor algorithms, not reducible to simple subtract. These stay as-is.
- Migrating `smoothAdd` — no sd-mecha equivalent. Stays as-is.
- Migrating elemental merging, quantized model handling, fine/adjust, exclude/include logic — unchanged.
- Adopting sd-mecha's model config / block naming system — supermerger's BLOCKID tables remain authoritative.
- Changing any UI layout beyond the calcmode dropdown.

---

## Calcmodes: In-Scope vs Out-of-Scope

| calcmode | migrated to sd-mecha? | reason if excluded |
|---|---|---|
| `normal` (Weight / Add / Triple / Twice) | Yes | Direct tensor math |
| `cosineA` | No | Different algorithm — would silently change user results |
| `cosineB` | No | Different algorithm — would silently change user results |
| `trainDifference` | No | Complex 3-tensor algorithm, not subtract |
| `extract` | No | Complex 3-tensor algorithm, not subtract |
| `smoothAdd` | No | No sd-mecha equivalent |
| `slerp` *(new)* | Yes | |
| `ties_sum` *(new)* | Yes | |
| `add_ties_with_dare` *(new)* | Yes | |
| `dropout` *(new)* | Yes | |

`methods.IN_SCOPE_CALCMODES` is a `frozenset` defined in `methods.py` listing all in-scope calcmodes. Out-of-scope calcmodes retain their existing code paths in `mergers.py` unchanged.

`uselerp` and `use32` flags currently affect `normal` mode only. sd-mecha's `weighted_sum` uses `torch.lerp` internally with float32 promotion — these flags are silently dropped for in-scope calcmodes. Out-of-scope calcmodes (which keep their existing code) retain the original flag behavior.

---

## Dependency Model

### Submodule

sd-mecha is added as a **git submodule** pinned to a specific commit:

```bash
git submodule add https://github.com/ljleb/sd-mecha scripts/sd_mecha
# then pin to a specific commit SHA via: git -C scripts/sd_mecha checkout <sha>
```

The submodule root is `scripts/sd_mecha/`, which contains the `sd_mecha/` Python package directory inside it. `methods.py` prepends `scripts/sd_mecha/` to `sys.path` so that `import sd_mecha` resolves to `scripts/sd_mecha/sd_mecha/`. Example:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'sd_mecha'))
import sd_mecha
from sd_mecha.extensions.builtin.merge_methods import linear as _mm_linear
from sd_mecha.extensions.builtin.merge_methods import ties as _mm_ties
```

### Additional pip dependencies

The following packages are added to the `requirements` list in `install.py`:

```python
"fuzzywuzzy",
"python-Levenshtein",
"scipy",
"PyYAML",
```

`fuzzywuzzy` is imported unconditionally at sd-mecha module load time; its absence causes `ImportError` before any merge can run. `scipy` is imported unconditionally in `ties.py`.

---

## Calling Convention for sd-mecha Functions

sd-mecha merge methods decorated with `@merge_method` return a `MergeRecipeNode` when called via their **public interface**. The underlying math is accessed via `.__wrapped__`.

Two categories exist:

**Category A — StateDict-typed** (`weighted_sum`, `add_difference`): their `__wrapped__` functions expect model arguments as `dict[str, Tensor]` and read `kwargs["key"]` to select the tensor. Use the `_call()` helper:

```python
def _call(fn, key, *tensors, **kwargs):
    """Wrap individual tensors as single-key dicts and invoke fn.__wrapped__."""
    wrapped = [{key: t} for t in tensors]
    return fn.__wrapped__(*wrapped, key=key, **kwargs)[key]
```

Alpha/beta values must be passed as `torch.tensor(value, dtype=torch.float32)`.

**Category B — bare Tensor** (`slerp`, `ties.dropout`, `ties.ties_sum`, `ties.ties_sum_with_dropout`): their `__wrapped__` functions accept `Tensor` arguments directly.

**Important — `probability` type for ties functions:** Although `ties_sum_with_dropout` and `ties_sum` annotate `probability` as `Parameter(Tensor)` in the decorator, the raw function bodies call `math.isclose(probability, 1.0)` directly. When calling `.__wrapped__` directly (bypassing the decorator), `probability` **must be passed as a plain Python `float`**, not `torch.tensor(...)`. Passing a Tensor raises `TypeError` from `math.isclose`.

**Category C — recipe-builder wrappers** (`wrappers.dropout`, `wrappers.add_ties_with_dare`): these are plain Python functions, not `@merge_method` decorated. They have **no `.__wrapped__` attribute**. They are never called directly in `methods.py`. Instead, `methods.py` replicates their logic using Category A/B primitives.

---

## New Files

### `scripts/mergers/methods.py` — Dispatch Layer

**Public interface:**

```python
IN_SCOPE_CALCMODES: frozenset[str]  # all calcmodes handled by this module

def dispatch(calcmode, mode, key, t0, t1, t2, alpha, beta) -> Tensor
```

- `t0`, `t1`, `t2`: `torch.Tensor` (individual key tensors, already loaded)
- `t2`: may be `None` for 2-model modes
- `alpha`, `beta`: Python `float` (resolved by caller via `resolve_alpha()`)
- Returns a `torch.Tensor` result for `key`

**Dispatch implementations:**

*`normal` — Weight mode:*
```python
return _call(_mm_linear.weighted_sum, key, t0, t1,
             alpha=torch.tensor(alpha, dtype=torch.float32))
```

*`normal` — Add mode* (t1 is already pre-subtracted B−C, passed in as `t1`):
```python
return _call(_mm_linear.add_difference, key, t0, t1,
             alpha=torch.tensor(alpha, dtype=torch.float32))
```

*`normal` — Triple mode* (A·(1-α-β) + B·α + C·β via two lerps):
```python
if alpha + beta != 0:
    mid = _call(_mm_linear.weighted_sum, key, t1, t2,
                alpha=torch.tensor(beta / (alpha + beta), dtype=torch.float32))
    return _call(_mm_linear.weighted_sum, key, t0, mid,
                 alpha=torch.tensor(alpha + beta, dtype=torch.float32))
else:
    return t0
```

*`normal` — Twice mode* (sequential: A→B then result→C):
```python
mid = _call(_mm_linear.weighted_sum, key, t0, t1,
            alpha=torch.tensor(alpha, dtype=torch.float32))
return _call(_mm_linear.weighted_sum, key, mid, t2,
             alpha=torch.tensor(beta, dtype=torch.float32))
```

*`slerp`:*
```python
return _mm_linear.slerp.__wrapped__(t0, t1,
           alpha=torch.tensor(alpha, dtype=torch.float32))
```

*`ties_sum`* (operates on deltas, not raw weights):
```python
delta = t1.float() - t0.float()
trimmed = _mm_ties.ties_sum.__wrapped__(delta,
              k=kwargs.get("k", 1.0),
              vote_sgn=kwargs.get("vote_sgn", False))
return (t0.float() + trimmed).to(t0.dtype)
```

*`dropout`* (uses `ties.dropout`, the `@merge_method` primitive — **not** `wrappers.dropout`):
```python
delta = t1.float() - t0.float()
dared = _mm_ties.dropout.__wrapped__(delta,
            probability=kwargs.get("probability", 0.9),   # plain float
            rescale=kwargs.get("rescale", 1.0),           # plain float
            seed=kwargs.get("seed", None))
return _call(_mm_linear.add_difference, key, t0, dared,
             alpha=torch.tensor(alpha, dtype=torch.float32))
```

Parameter semantics: `alpha` is supermerger's standard per-block blend factor (from MBW), applied via `add_difference` as the delta scale. `rescale` is `ties.dropout`'s own scaling divisor (default `1.0` = no rescale). `probability` is the DARE dropout keep-probability (default `0.9`). This surface differs from `wrappers.dropout` where `alpha` controls the add_difference blend and `rescale` is not separately exposed.

*`add_ties_with_dare`* (3-step chain mirroring `wrappers.add_ties_with_dare`):
```python
# Step 1: delta
delta = t1.float() - t0.float()
# Step 2: ties_sum_with_dropout (DARE dropout + TIES trim combined)
result_delta = _mm_ties.ties_sum_with_dropout.__wrapped__(
    delta,
    probability=kwargs.get("probability", 0.9),
    della_eps=kwargs.get("della_eps", 0.0),
    rescale=kwargs.get("rescale", True),
    k=kwargs.get("k", 1.0),
    vote_sgn=kwargs.get("vote_sgn", False),
    apply_stock=kwargs.get("apply_stock", False),
    cos_eps=kwargs.get("cos_eps", 1e-6),
    apply_median=kwargs.get("apply_median", False),
    eps=kwargs.get("eps", 1e-6),
    maxiter=kwargs.get("maxiter", 100),
    ftol=kwargs.get("ftol", 1e-20),
    seed=kwargs.get("seed", None),
)
# Step 3: add back to base with alpha scale
return _call(_mm_linear.add_difference, key, t0, result_delta,
             alpha=torch.tensor(alpha, dtype=torch.float32))
```

**Error handling:**
- Unknown `calcmode`: `raise ValueError(f"Unknown calcmode for methods.dispatch: {calcmode}")`
- `t2 is None` when Triple/Twice: `raise ValueError(f"calcmode '{calcmode}' in mode '{mode}' requires t2 (model C)")`
- Tensor shape/dtype errors: let sd-mecha raise naturally.

### `scripts/mergers/streamer.py` — Streaming Save Path

Called by `smerge()` **after the preamble** (after architecture detection, dtype normalization, Stage 0 pre-subtraction for Add mode, cosine pre-computation) and before the Stage 1/2 key loop, when the user has requested a save.

**Function signature:**

```python
def merge_and_save(save_path, theta_0, theta_1, theta_2,
                   calcmode, mode, key_params, ...)
```

`key_params` is a named tuple or dict carrying all parameters needed by `resolve_alpha()` and `methods.dispatch()`.

**Flow:**

1. Open output `.safetensors` file for streaming writes.
2. For each key in `theta_0`:
   a. Call `resolve_alpha(key, ...)` → `(current_alpha, current_beta, skip)`.
   b. If `skip`: write `theta_0[key]` to output unchanged; continue.
   c. Handle inpaint slice: if `theta_0[key].shape != theta_1[key].shape` and `a[0:1]+a[2:] == b[0:1]+b[2:]`, operate on `t0 = theta_0[key][:, 0:4, :, :]`, write result into a clone of `theta_0[key]` before writing to output.
   d. Call `methods.dispatch(calcmode, mode, key, t0, theta_1[key], theta_2.get(key) if theta_2 else None, current_alpha, current_beta)`.
   e. Write result tensor to output file immediately; discard.
3. **Stage 2/2** — copy keys in `theta_1` absent from `theta_0` (text encoder keys etc.) directly to output.
4. Finalize and close output.

**Memory note:** `theta_0`, `theta_1`, `theta_2` are still fully loaded through the preamble. The streaming benefit is that the **result dict** is never held in memory — keys are written and discarded. Full lazy input loading (true key-by-key disk reads) is out of scope for this migration.

---

## Modified Files

### `scripts/mergers/mergers.py`

**Change 1: 2-model calcmode validation at top of `smerge()`**

Before any model loading, add:

```python
TWO_MODEL_ONLY = {"slerp", "ties_sum", "add_ties_with_dare", "dropout"}
if calcmode in TWO_MODEL_ONLY and mode in (MODES[2], MODES[3]):  # Triple or Twice
    return f"ERROR: calcmode '{calcmode}' only supports Weight or Add mode.", *NON4
# Add mode with 3 models: the preamble's Stage 0 pre-subtracts theta_2 from theta_1
# and stores the result back into theta_1, so by the time the key loop (or streamer)
# runs, t1 already holds (B - C) and t2 is None / not passed to dispatch.
# New calcmodes therefore work correctly in Add mode — theta_2 is consumed upstream.
```

This prevents loading three models before discovering the incompatibility.

**Change 2: Extract `resolve_alpha()` helper**

Extracted from the key loop into a standalone function:

```python
def resolve_alpha(key, isxl, isflux, useblocks, usebeta,
                  base_alpha, base_beta, weights_a, weights_b,
                  deep, randomer, num, lucks, deepprint, esettings,
                  inex, ex_blocks, ex_elems,
                  weights_a_excluded, weights_b_excluded)
    -> (current_alpha: float, current_beta: float, skip: bool)
```

Encapsulates: `blockfromkey()`, BLOCKID index lookup, MBW weight lookup, `elementals()`, `excluder()`. Mutates `weights_a_excluded`/`weights_b_excluded` in place (for logging). Returns three values consistently throughout.

**Change 3: Replace math block with `methods.dispatch()`**

In the key loop, the large `if calcmode == "normal": ... elif ...` chain is replaced:

```python
current_alpha, current_beta, skip = resolve_alpha(key, ...)
if skip:
    continue

# inpaint slice handling
a, b = list(theta_0[key].shape), list(theta_1[key].shape)
assert_inpaint(a, b, key)
if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
    slice_only = True
    t0 = theta_0[key][:, 0:4, :, :]
else:
    slice_only = False
    t0 = theta_0[key]

if calcmode in methods.IN_SCOPE_CALCMODES:
    result = methods.dispatch(calcmode, mode, key, t0, theta_1[key],
                              theta_2[key] if theta_2 is not None else None,
                              current_alpha, current_beta)
    if slice_only:
        theta_0[key][:, 0:4, :, :] = result
    else:
        theta_0[key] = result

elif calcmode == "cosineA":
    # existing code unchanged
    ...
elif calcmode == "trainDifference":
    # existing code unchanged
    ...
# etc.
```

**Change 4: Route to `streamer` after preamble**

The save path is derived inside `smerge()` from existing data: check `SAVEMODES[0] in save_sets` (the "save model" flag), then derive the output filepath using `makemodelname()` + the model directory (the same logic already used in `smergegen()`). No new parameter to `smerge()` is required.

After all preamble steps and before the Stage 1/2 key loop:

```python
save_path = None
if SAVEMODES[0] in save_sets or SAVEMODES[1] in save_sets:
    save_path = <derive filepath from custom_name / makemodelname() as in smergegen()>

if save_path:
    return streamer.merge_and_save(save_path, theta_0, theta_1, theta_2, ...)
```

### `scripts/supermerger.py`

Add four entries to the calcmode dropdown / `CALCMODES` list:

```python
"slerp", "ties_sum", "add_ties_with_dare", "dropout"
```

No other UI changes.

---

## Shared Helper: `resolve_alpha()`

**Parameters** (all sourced from `smerge()` locals):

`key, isxl, isflux, useblocks, usebeta, base_alpha, base_beta, weights_a, weights_b, deep, randomer, num, lucks, deepprint, esettings, inex, ex_blocks, ex_elems, weights_a_excluded, weights_b_excluded`

**Returns:** `(current_alpha: float, current_beta: float, skip: bool)`

Encapsulates (extracted verbatim from the existing key loop):
- `blockfromkey(key, isxl, isflux)` → block, blocks26
- BLOCKID / BLOCKIDXLL / BLOCKIDFLUX index lookup
- MBW weight selection with `len(weights_a) == 109 or "use extended XL" in esettings` check
- `elementals()` call for elemental overrides
- `excluder()` for include/exclude logic
- Side-effect mutation of `weights_a_excluded` / `weights_b_excluded` lists

---

## Data Flow

### In-Memory Path (no save)

```
smerge() preamble
  └─ for each key in theta_0:
       ├─ resolve_alpha()          [supermerger's block system]
       ├─ methods.dispatch()       [sd-mecha math, for in-scope calcmodes]
       │    └─ returns Tensor
       └─ theta_0[key] updated
  └─ Stage 2/2: copy theta_1-only keys into theta_0
  └─ load merged dict into running model
```

### Save-to-Disk Path

```
smerge() preamble
  └─ save_path derived from save_sets + custom_name
  └─ streamer.merge_and_save()
       ├─ open output for streaming writes
       ├─ for each key in theta_0:
       │    ├─ resolve_alpha()     [same helper]
       │    ├─ methods.dispatch()  [same math]
       │    └─ write key to output, discard tensor
       └─ Stage 2/2: copy theta_1-only keys to output
```

---

## Testing Approach

- Merge two SD1.5 models, Weight mode, no save → output within float32 tolerance of pre-migration.
- Merge two SD1.5 models, Weight mode, save to disk → file output matches in-memory result tensor-for-tensor.
- MBW enabled on both paths → per-block alphas apply correctly.
- Inpainting model merge → slice handling preserves correct channel count on both paths.
- Out-of-scope calcmodes (`cosineA`, `trainDifference`, etc.) → bit-identical output to pre-migration.
- Each new calcmode (slerp, ties_sum, add_ties_with_dare, dropout) → no crash; output is a valid loadable model.
- Triple/Twice mode + new calcmode at top of `smerge()` → error returned before model loading.
- SDXL and Flux models on both paths.

---

## Future Extension

To add another sd-mecha method later:
1. Add one function in `methods.py`'s dispatch.
2. Add its name to `IN_SCOPE_CALCMODES`.
3. Add one entry to the calcmode dropdown in `supermerger.py`.
4. Update submodule commit if the method requires a newer sd-mecha release.
