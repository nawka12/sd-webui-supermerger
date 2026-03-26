# sd-mecha Backend Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace supermerger's per-tensor math with sd-mecha's implementations for all supported calcmodes, add four new calcmodes (slerp, ties_sum, add_ties_with_dare, dropout), and add a memory-efficient streaming save path.

**Architecture:** sd-mecha is vendored as a git submodule at `scripts/sd_mecha/`. A new `methods.py` dispatch layer maps calcmode names to sd-mecha functions called via `.__wrapped__`. A new `streamer.py` handles memory-efficient save-to-disk by writing keys one at a time. `mergers.py` orchestration (block assignment, MBW, model loading) is unchanged; only the per-tensor math calls at the bottom of the key loop are replaced.

**Tech Stack:** Python, PyTorch, safetensors, sd-mecha (submodule), existing A1111/Forge WebUI environment.

**Spec:** `docs/superpowers/specs/2026-03-26-sd-mecha-backend-migration-design.md`

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/sd_mecha/` | Create (submodule) | sd-mecha library, pinned commit |
| `scripts/mergers/methods.py` | Create | Dispatch layer: calcmode → sd-mecha math |
| `scripts/mergers/streamer.py` | Create | Save path: builds output dict key-by-key, writes once via safetensors `save_file()` |
| `scripts/mergers/mergers.py` | Modify | Extract `resolve_alpha()`, replace dispatch block, remove `precosine` pre-pass, add save routing, add 2-model validation |
| `scripts/supermerger.py` | Modify | Add 4 new calcmode entries to `CALCMODES` list (line 42) |
| `install.py` | Modify | Add `fuzzywuzzy`, `python-Levenshtein`, `scipy`, `PyYAML` to requirements |
| `tests/conftest.py` | Create | Mock A1111 WebUI modules so unit tests run outside WebUI |
| `tests/test_methods.py` | Create | Unit tests for every `methods.dispatch()` branch |
| `tests/test_streamer.py` | Create | Unit tests for `streamer.merge_and_save()` |
| `tests/test_resolve_alpha.py` | Create | Unit tests for `resolve_alpha()` helper |

---

## Task 1: Add sd-mecha submodule and pip dependencies

**Files:**
- Create: `scripts/sd_mecha/` (git submodule)
- Modify: `install.py`

- [ ] **Step 1: Add the submodule**

```bash
cd /mnt/a/stable-diffusion-webui-reForge/extensions/sd-webui-supermerger
git submodule add https://github.com/ljleb/sd-mecha scripts/sd_mecha
```

Expected: `scripts/sd_mecha/` directory created, `.gitmodules` file updated.

- [ ] **Step 2: Pin to a specific commit**

Check the latest stable commit SHA from sd-mecha's main branch and pin it:

```bash
cd scripts/sd_mecha
git log --oneline -5       # pick a stable commit
git checkout <chosen-sha>
cd ../..
git add scripts/sd_mecha .gitmodules
```

- [ ] **Step 3: Verify the submodule structure**

```bash
ls scripts/sd_mecha/sd_mecha/
```

Expected: a Python package directory containing `__init__.py` and subdirectories including `extensions/`.

- [ ] **Step 4: Verify sd-mecha is importable from the submodule**

```bash
python3 -c "
import sys
sys.path.insert(0, 'scripts/sd_mecha')
import sd_mecha
print('sd_mecha imported OK')
from sd_mecha.extensions.builtin.merge_methods import linear as l
print('linear imported OK:', dir(l))
"
```

Expected: prints both OK lines. If `ImportError` on `fuzzywuzzy` or `scipy`, proceed to step 5 first.

- [ ] **Step 5: Add missing pip dependencies to `install.py`**

Open `install.py`. The `requirements` list currently reads:
```python
requirements = [
"diffusers==0.31.0",
"scikit-learn",
"accelerate"
]
```

Add the four new entries:
```python
requirements = [
"diffusers==0.31.0",
"scikit-learn",
"accelerate",
"fuzzywuzzy",
"python-Levenshtein",
"scipy",
"PyYAML",
]
```

- [ ] **Step 6: Install the new dependencies and re-verify import**

```bash
pip install fuzzywuzzy python-Levenshtein scipy PyYAML
python3 -c "
import sys
sys.path.insert(0, 'scripts/sd_mecha')
from sd_mecha.extensions.builtin.merge_methods import linear as _mm_linear
from sd_mecha.extensions.builtin.merge_methods import ties as _mm_ties
from sd_mecha.extensions.builtin.merge_methods import cosine as _mm_cosine
print('all imports OK')
print('weighted_sum.__wrapped__:', _mm_linear.weighted_sum.__wrapped__)
print('slerp.__wrapped__:', _mm_linear.slerp.__wrapped__)
print('ties_sum.__wrapped__:', _mm_ties.ties_sum.__wrapped__)
print('add_cosine_a.__wrapped__:', _mm_cosine.add_cosine_a.__wrapped__)
"
```

Expected: all lines print without error. Confirm `__wrapped__` attributes exist.

- [ ] **Step 7: Verify cosine function argument types**

```bash
python3 -c "
import sys, inspect
sys.path.insert(0, 'scripts/sd_mecha')
from sd_mecha.extensions.builtin.merge_methods import cosine as _mm_cosine
print(inspect.getsource(_mm_cosine.add_cosine_a.__wrapped__))
print(inspect.getsource(_mm_cosine.add_cosine_b.__wrapped__))
"
```

Read the output. Confirm whether `add_cosine_a.__wrapped__` / `add_cosine_b.__wrapped__` take bare `Tensor` arguments or `StateDict[Tensor]` arguments. If they take bare tensors, they are Category B (called directly). If they take StateDicts, they need `_call()`. Update the dispatch table in Task 3 accordingly.

- [ ] **Step 8: Commit**

```bash
git add scripts/sd_mecha .gitmodules install.py
git commit -m "feat: add sd-mecha as git submodule and install pip dependencies"
```

---

## Task 2: Create test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

Unit tests for `methods.py` and `streamer.py` need to run outside the A1111 WebUI environment. `mergers.py` imports many A1111 modules at the top level; without mocking them, importing anything from `mergers.py` (including `resolve_alpha()`) fails immediately.

- [ ] **Step 1: Create the tests package**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Write `tests/conftest.py`**

```python
# tests/conftest.py
"""
Mock A1111/Forge WebUI modules so tests can import from scripts/ without
needing a running WebUI environment.
"""
import sys
import types
import os
import pytest

# --- Add sd_mecha submodule to path ---
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_repo_root, "scripts", "sd_mecha"))
sys.path.insert(0, os.path.join(_repo_root, "scripts"))

def _make_mock_module(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod

# Minimal stubs required by mergers.py imports
_modules_stub = _make_mock_module("modules",
    shared=_make_mock_module("modules.shared", opts=object(), cmd_opts=object()),
    processing=_make_mock_module("modules.processing"),
    sd_models=_make_mock_module("modules.sd_models", model_path="/tmp"),
    sd_vae=_make_mock_module("modules.sd_vae", vae_dict={}),
    images=_make_mock_module("modules.images"),
    sd_samplers=_make_mock_module("modules.sd_samplers"),
    scripts=_make_mock_module("modules.scripts"),
    devices=_make_mock_module("modules.devices", torch_gc=lambda: None),
    extras=_make_mock_module("modules.extras"),
    script_callbacks=_make_mock_module("modules.script_callbacks"),
)
_modules_stub.ui = _make_mock_module("modules.ui", plaintext_to_html=lambda x: x)

for mod_name, mod in [
    ("modules", _modules_stub),
    ("modules.shared", _modules_stub.shared),
    ("modules.processing", _modules_stub.processing),
    ("modules.sd_models", _modules_stub.sd_models),
    ("modules.sd_vae", _modules_stub.sd_vae),
    ("modules.images", _modules_stub.images),
    ("modules.sd_samplers", _modules_stub.sd_samplers),
    ("modules.scripts", _modules_stub.scripts),
    ("modules.devices", _modules_stub.devices),
    ("modules.extras", _modules_stub.extras),
    ("modules.script_callbacks", _modules_stub.script_callbacks),
    ("modules.ui", _modules_stub.ui),
    ("launch", _make_mock_module("launch", git_tag=lambda: "test")),
]:
    sys.modules.setdefault(mod_name, mod)

# backend stubs for Forge
for name in ["backend", "backend.memory_management", "backend.utils"]:
    sys.modules.setdefault(name, _make_mock_module(name,
        load_torch_file=lambda *a, **kw: {}))
```

- [ ] **Step 3: Verify conftest loads without error**

```bash
cd /mnt/a/stable-diffusion-webui-reForge/extensions/sd-webui-supermerger
python3 -c "import tests.conftest; print('conftest OK')"
```

Expected: `conftest OK`

- [ ] **Step 4: Commit**

```bash
git add tests/
git commit -m "test: add test infrastructure and A1111 module mocks"
```

---

## Task 3: Create `scripts/mergers/methods.py` (TDD)

**Files:**
- Create: `tests/test_methods.py`
- Create: `scripts/mergers/methods.py`

**Mode string convention:** `dispatch()` receives the full UI mode string (e.g. `"Weight sum"`, `"Add difference"`, `"Triple sum"`, `"sum Twice"`). The mode branches use substring matching (`"Add" in mode`, `"Triple" in mode`, `"Twice" in mode`) matching how `mergers.py` uses its `MODES` list. Tests below use these full strings.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_methods.py
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'sd_mecha'))

from mergers import methods

KEY = "model.diffusion_model.input_blocks.0.0.weight"

# Real UI mode strings (substring-matched inside dispatch)
MODE_WEIGHT = "Weight sum"
MODE_ADD    = "Add difference"
MODE_TRIPLE = "Triple sum"
MODE_TWICE  = "sum Twice"

def t(val=0.5, shape=(4, 4)):
    """Helper: float32 tensor filled with val."""
    return torch.full(shape, val, dtype=torch.float32)


class TestInScopeSet:
    def test_contains_all_expected_modes(self):
        expected = {"normal", "cosineA", "cosineB", "slerp",
                    "ties_sum", "add_ties_with_dare", "dropout"}
        assert expected.issubset(methods.IN_SCOPE_CALCMODES)


class TestDispatchNormal:
    def test_weight_mode_alpha_0_returns_t0(self):
        t0, t1 = t(1.0), t(2.0)
        result = methods.dispatch("normal", MODE_WEIGHT, KEY, t0, t1, None, 0.0, 0.0)
        assert torch.allclose(result, t0)

    def test_weight_mode_alpha_1_returns_t1(self):
        t0, t1 = t(1.0), t(2.0)
        result = methods.dispatch("normal", MODE_WEIGHT, KEY, t0, t1, None, 1.0, 0.0)
        assert torch.allclose(result, t1)

    def test_weight_mode_alpha_half_is_midpoint(self):
        t0, t1 = t(0.0), t(2.0)
        result = methods.dispatch("normal", MODE_WEIGHT, KEY, t0, t1, None, 0.5, 0.0)
        assert torch.allclose(result, t(1.0), atol=1e-5)

    def test_add_mode_applies_delta(self):
        # In Add mode, t1 is already (B - C). result = t0 + alpha * t1
        t0, t1 = t(1.0), t(0.5)
        result = methods.dispatch("normal", MODE_ADD, KEY, t0, t1, None, 1.0, 0.0)
        assert torch.allclose(result, t(1.5), atol=1e-5)

    def test_triple_mode_equal_weights(self):
        # alpha=beta=0: returns t0
        t0, t1, t2 = t(1.0), t(2.0), t(3.0)
        result = methods.dispatch("normal", MODE_TRIPLE, KEY, t0, t1, t2, 0.0, 0.0)
        assert torch.allclose(result, t0)

    def test_triple_mode_raises_if_t2_none(self):
        with pytest.raises(ValueError, match="requires t2"):
            methods.dispatch("normal", MODE_TRIPLE, KEY, t(1.0), t(2.0), None, 0.5, 0.3)

    def test_twice_mode_sequential(self):
        # alpha=1, beta=0: result = t1; then lerp(t1, t2, 0) = t1
        t0, t1, t2 = t(0.0), t(1.0), t(2.0)
        result = methods.dispatch("normal", MODE_TWICE, KEY, t0, t1, t2, 1.0, 0.0)
        assert torch.allclose(result, t1, atol=1e-5)

    def test_unknown_calcmode_raises(self):
        with pytest.raises(ValueError, match="Unknown calcmode"):
            methods.dispatch("nonexistent", MODE_WEIGHT, KEY, t(0.0), t(1.0), None, 0.5, 0.0)


class TestDispatchCosine:
    def test_cosineA_returns_tensor_same_shape(self):
        t0, t1 = t(0.3), t(0.7)
        result = methods.dispatch("cosineA", "Weight", KEY, t0, t1, None, 0.5, 0.0)
        assert result.shape == t0.shape

    def test_cosineB_returns_tensor_same_shape(self):
        t0, t1 = t(0.3), t(0.7)
        result = methods.dispatch("cosineB", "Weight", KEY, t0, t1, None, 0.5, 0.0)
        assert result.shape == t0.shape


class TestDispatchSlerp:
    def test_slerp_alpha_0_returns_t0(self):
        t0 = torch.randn(4, 4)
        t1 = torch.randn(4, 4)
        result = methods.dispatch("slerp", "Weight", KEY, t0, t1, None, 0.0, 0.0)
        assert torch.allclose(result.float(), t0.float(), atol=1e-4)

    def test_slerp_alpha_1_returns_t1(self):
        t0 = torch.randn(4, 4)
        t1 = torch.randn(4, 4)
        result = methods.dispatch("slerp", "Weight", KEY, t0, t1, None, 1.0, 0.0)
        assert torch.allclose(result.float(), t1.float(), atol=1e-4)


class TestDispatchTiesSum:
    def test_ties_sum_same_tensors_returns_t0(self):
        t0 = torch.randn(8, 8)
        result = methods.dispatch("ties_sum", "Weight", KEY, t0, t0.clone(), None, 1.0, 0.0)
        # delta is zero, trimmed is zero, result is t0
        assert torch.allclose(result.float(), t0.float(), atol=1e-5)

    def test_ties_sum_returns_correct_shape(self):
        t0, t1 = torch.randn(8, 8), torch.randn(8, 8)
        result = methods.dispatch("ties_sum", "Weight", KEY, t0, t1, None, 1.0, 0.0)
        assert result.shape == t0.shape


class TestDispatchDropout:
    def test_dropout_returns_correct_shape(self):
        t0, t1 = torch.randn(8, 8), torch.randn(8, 8)
        result = methods.dispatch("dropout", "Weight", KEY, t0, t1, None, 0.5, 0.0)
        assert result.shape == t0.shape


class TestDispatchAddTiesWithDare:
    def test_add_ties_with_dare_returns_correct_shape(self):
        t0, t1 = torch.randn(8, 8), torch.randn(8, 8)
        result = methods.dispatch("add_ties_with_dare", "Weight", KEY, t0, t1, None, 0.5, 0.0)
        assert result.shape == t0.shape
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
cd /mnt/a/stable-diffusion-webui-reForge/extensions/sd-webui-supermerger
python3 -m pytest tests/test_methods.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` — `methods` does not exist yet.

- [ ] **Step 3: Create `scripts/mergers/methods.py`**

```python
# scripts/mergers/methods.py
"""
Dispatch layer: maps supermerger calcmodes to sd-mecha math functions.

Calling conventions:
  Category A (StateDict-typed): weighted_sum, add_difference
    → use _call(fn, key, *tensors, **kwargs) wrapper
  Category B (bare Tensor): slerp, cosine, ties, dropout primitives
    → call fn.__wrapped__() directly
  Category C (recipe builders): wrappers.dropout, wrappers.add_ties_with_dare
    → NOT called directly; their chains are replicated here using A/B primitives
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
    """
    wrapped = [{key: t} for t in tensors]
    return fn.__wrapped__(*wrapped, key=key, **kwargs)[key]


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
    # No import from mergers.py — avoids circular imports.
    _MODE_ADD    = "Add"
    _MODE_TRIPLE = "Triple"
    _MODE_TWICE  = "Twice"

    # ---- normal mode (Weight / Add / Triple / Twice) ----
    if calcmode == "normal":
        if _MODE_ADD in mode:
            return _call(_mm_linear.add_difference, key, t0, t1, alpha=a)

        if _MODE_TRIPLE in mode:  # Triple: A*(1-α-β) + B*α + C*β
            if t2 is None:
                raise ValueError(f"calcmode 'normal' in mode '{mode}' requires t2 (model C)")
            if alpha + beta != 0:
                ratio = torch.tensor(beta / (alpha + beta), dtype=torch.float32)
                mid = _call(_mm_linear.weighted_sum, key, t1, t2, alpha=ratio)
                return _call(_mm_linear.weighted_sum, key, t0, mid,
                             alpha=torch.tensor(alpha + beta, dtype=torch.float32))
            return t0

        if _MODE_TWICE in mode:  # Twice: (A→B) then (result→C)
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
        # probability must be plain float — math.isclose() is called on it internally
        result_delta = _mm_ties.ties_sum_with_dropout.__wrapped__(
            delta,
            probability=0.9,   # plain float
            della_eps=0.0,
            rescale=True,
            k=1.0,
            vote_sgn=False,
            apply_stock=False,
            cos_eps=1e-6,
            apply_median=False,
            eps=1e-6,
            maxiter=100,
            ftol=1e-20,
            seed=None,
        )
        return _call(_mm_linear.add_difference, key, t0, result_delta, alpha=a)

    raise ValueError(f"Unknown calcmode for methods.dispatch: {calcmode!r}")
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_methods.py -v
```

Expected: all tests pass. If any fail, fix the dispatch implementations before continuing. Pay attention to:
- `_COSINE_IS_STATEDICT` detection — if cosine tests fail with subscription errors, the detection logic needs adjustment.
- Shape/dtype mismatches in ties/dropout tests.

- [ ] **Step 5: Commit**

```bash
git add scripts/mergers/methods.py tests/test_methods.py
git commit -m "feat: add sd-mecha dispatch layer (methods.py)"
```

---

## Task 4: Extract `resolve_alpha()` from `mergers.py` (TDD)

**Files:**
- Create: `tests/test_resolve_alpha.py`
- Modify: `scripts/mergers/mergers.py` (lines 432–481)

`resolve_alpha()` currently lives inline in `smerge()`'s key loop (lines 432–481). Extract it as a module-level function so both the in-memory loop and `streamer.py` can call it.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_resolve_alpha.py
import pytest
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

# conftest.py already installed A1111 mocks via pytest auto-discovery
from mergers.mergers import resolve_alpha, BLOCKID, BLOCKIDXLL


class TestResolveAlphaSd15:
    """Tests using SD 1.5 block IDs (26 blocks)."""

    # Build minimal weights_a with 26 values (BASE + 25 blocks)
    W26 = [float(i) / 25 for i in range(26)]

    def test_base_block_uses_base_alpha_when_not_useblocks(self):
        wa_excluded = [None] * 110
        wb_excluded = [None] * 110
        alpha, beta, skip = resolve_alpha(
            key="model.diffusion_model.input_blocks.0.0.weight",
            isxl=False, isflux=False,
            useblocks=False, usebeta=False,
            base_alpha=0.42, base_beta=0.0,
            weights_a=[], weights_b=[],
            deep=[], randomer=np.zeros(3000), num=0,
            lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Off", ex_blocks=[], ex_elems=[""],
            weights_a_excluded=wa_excluded,
            weights_b_excluded=wb_excluded,
        )
        assert alpha == 0.42
        assert skip is False

    def test_mbw_applies_per_block_alpha(self):
        wa_excluded = [None] * 110
        wb_excluded = [None] * 110
        # 26 weights: [BASE=0.0, IN00=1/25, IN01=2/25, ...]
        # The key below is in input_blocks.1 → IN01 → BLOCKID index 2 → weights_a[1] = 1/25
        weights_a = [0.0] + [float(i) / 25 for i in range(25)]
        alpha, beta, skip = resolve_alpha(
            key="model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_q.weight",
            isxl=False, isflux=False,
            useblocks=True, usebeta=False,
            base_alpha=0.5, base_beta=0.0,
            weights_a=weights_a, weights_b=[0.0] * 26,
            deep=[], randomer=np.zeros(3000), num=0,
            lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Off", ex_blocks=[], ex_elems=[""],
            weights_a_excluded=wa_excluded,
            weights_b_excluded=wb_excluded,
        )
        # input_blocks.1 maps to IN01 = BLOCKID index 2; weights_a[index-1] = weights_a[1] = 1/25
        # Verify per-block alpha was applied (not base_alpha=0.5)
        assert skip is False
        assert abs(alpha - (1.0 / 25.0)) < 1e-6, (
            f"Expected per-block alpha={1/25:.6f}, got {alpha}. "
            "Check BLOCKID table to confirm IN01 index and adjust assertion if needed."
        )

    def test_excluded_block_returns_skip_true(self):
        wa_excluded = [None] * 110
        wb_excluded = [None] * 110
        weights_a = [0.5] * 26
        alpha, beta, skip = resolve_alpha(
            key="model.diffusion_model.input_blocks.0.0.weight",
            isxl=False, isflux=False,
            useblocks=True, usebeta=False,
            base_alpha=0.5, base_beta=0.0,
            weights_a=weights_a, weights_b=[0.0] * 26,
            deep=[], randomer=np.zeros(3000), num=0,
            lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Exclude", ex_blocks=["IN00"], ex_elems=[""],
            weights_a_excluded=wa_excluded,
            weights_b_excluded=wb_excluded,
        )
        assert skip is True
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
python3 -m pytest tests/test_resolve_alpha.py -v 2>&1 | head -20
```

Expected: `ImportError` — `resolve_alpha` does not exist yet.

- [ ] **Step 3: Extract `resolve_alpha()` in `mergers.py`**

Locate the block assignment and MBW logic at lines 432–481 of `mergers.py`. Extract it into a new module-level function immediately above `smerge()` (around line 219).

**Bug fix during extraction:** Line 469 of the original `mergers.py` reads `current_alpha = weights_b[weight_index_xl - 1]` — this assigns to `current_alpha` instead of `current_beta`, which is a pre-existing bug. The extracted `resolve_alpha()` below corrects this to `current_beta = weights_b[weight_index_xl - 1]`. This is an intentional behavior fix, not a verbatim extraction.

```python
def resolve_alpha(key, isxl, isflux, useblocks, usebeta,
                  base_alpha, base_beta, weights_a, weights_b,
                  deep, randomer, num, lucks, deepprint, esettings,
                  inex, ex_blocks, ex_elems,
                  weights_a_excluded, weights_b_excluded):
    """
    Resolve per-key alpha and beta from block assignment and MBW weights.
    Mutates weights_a_excluded / weights_b_excluded in place for logging.
    Returns (current_alpha, current_beta, skip).
    """
    weight_index = -1
    current_alpha = base_alpha
    current_beta = base_beta

    block, blocks26 = blockfromkey(key, isxl, isflux)

    skip = (inex != "Off"
            and (ex_blocks or (ex_elems != [""]))
            and excluder(block, blocks26, inex, ex_blocks, ex_elems, key))

    if isflux and blocks26 in BLOCKIDFLUX:
        weight_index = BLOCKIDFLUX.index(blocks26)
    elif isxl and blocks26 in BLOCKIDXLL:
        weight_index = BLOCKIDXLL.index(blocks26)
    elif blocks26 in BLOCKID:
        weight_index = BLOCKID.index(blocks26)
    else:
        return current_alpha, current_beta, True  # key not in any block → skip

    weight_index_xl = BLOCKIDXLLL.index(block)

    if useblocks:
        if weight_index > 0:
            if skip: weights_a_excluded[weight_index] = 0
            current_alpha = weights_a[weight_index - 1]
            if len(weights_a) == 109:
                if skip: weights_a_excluded[weight_index_xl] = 0
                current_alpha = weights_a[weight_index_xl - 1]

            if usebeta:
                if skip: weights_b_excluded[weight_index] = 0
                current_beta = weights_b[weight_index - 1]
                if len(weights_b) == 109:
                    if skip: weights_b_excluded[weight_index_xl] = 0
                    current_beta = weights_b[weight_index_xl - 1]

        if weight_index == 0:
            if len(weights_a) == 109 and weight_index_xl == 1:
                if skip: weights_a_excluded[weight_index_xl] = 0
                current_alpha = weights_a[weight_index_xl - 1]
            if len(weights_b) == 109 and usebeta and weight_index_xl == 1:
                if skip: weights_b_excluded[weight_index_xl] = 0
                current_beta = weights_b[weight_index_xl - 1]

            if skip: weights_a_excluded[0] = 0
            if skip: weights_b_excluded[0] = 0

    if skip:
        return current_alpha, current_beta, True

    if len(deep) > 0:
        current_alpha = elementals(
            key, weight_index, weight_index_xl, deep, randomer, num,
            lucks, deepprint, current_alpha,
            len(weights_a) == 109 or "use extended XL" in esettings)

    return current_alpha, current_beta, False
```

Then in the key loop, replace lines 432–481 with:

```python
current_alpha, current_beta, skip = resolve_alpha(
    key, isxl, isflux, useblocks, usebeta,
    base_alpha, base_beta, weights_a, weights_b,
    deep, randomer, num, lucks, deepprint, esettings,
    inex, ex_blocks, ex_elems,
    weights_a_excluded, weights_b_excluded,
)
if skip:
    continue
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_resolve_alpha.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/mergers/mergers.py tests/test_resolve_alpha.py
git commit -m "refactor: extract resolve_alpha() helper from smerge key loop"
```

---

## Task 5: Replace calcmode dispatch in `mergers.py` key loop

**Files:**
- Modify: `scripts/mergers/mergers.py` (lines ~409–411, ~490–552)

This task has no new tests — the change is verified by confirming the existing manual test case (Task 8) plus a quick sanity import.

- [ ] **Step 1: Remove the `precosine` pre-pass**

Locate lines 409–411:
```python
##### Stage 0/2 in Cosine
if "cosine" in calcmode:
    sim, sims = precosine("A" in calcmode,theta_0,theta_1)
```

Remove all three lines. The `sim` and `sims` variables are only used in `cosine()` calls at line 551, which will also be removed.

- [ ] **Step 2: Add `import` for methods at the top of `mergers.py`**

After the existing local imports (around line 37–38). Use the same import style as existing imports in `mergers.py` (line 31: `from scripts.mergers.model_util import ...`):
```python
from scripts.mergers import methods as _merge_methods
from scripts.mergers import streamer as _streamer
```

If running tests via `conftest.py` (which inserts `scripts/` into `sys.path`), this import resolves correctly because conftest inserts the **repo root** first (`scripts/` as a package) and `scripts/` as a flat path. The absolute form `from scripts.mergers import ...` requires the repo root to be on `sys.path`, which the WebUI provides automatically. The conftest must also insert the repo root (parent of `scripts/`); verify this is the case.

- [ ] **Step 3: Add 2-model validation after format check (around line 307)**

Locate the format check block:
```python
#format check
if model_a =="" or model_b =="" or ((not MODES[0] in mode) and model_c=="") :
    return "ERROR: Necessary model is not selected",*NON4
```

Add immediately after it:
```python
if calcmode in _merge_methods.TWO_MODEL_ONLY and (MODES[2] in mode or MODES[3] in mode):
    return f"ERROR: calcmode '{calcmode}' only supports Weight or Add mode.", *NON4
```

- [ ] **Step 4: Replace the calcmode dispatch block in the key loop**

Locate the dispatch block starting at line ~490:
```python
if calcmode == "normal":
    if a != b and a[0:1] + a[2:] == b[0:1] + b[2:]:
        ...
    # (lines 490–544)

elif "cosine" in calcmode:
    if "first_stage_model" in key: continue
    cosine(calcmode,key,sim,sims,current_alpha,theta_0,theta_1,num,block,uselerp)
    # (lines 549–551)
```

Replace both `if calcmode == "normal":` and `elif "cosine" in calcmode:` blocks with:

```python
if calcmode in _merge_methods.IN_SCOPE_CALCMODES:
    a_shape = list(theta_0[key].shape)
    b_shape = list(theta_1[key].shape)
    if a_shape != b_shape and a_shape[0:1] + a_shape[2:] == b_shape[0:1] + b_shape[2:]:
        slice_only = True
        t0 = theta_0[key][:, 0:4, :, :]
    else:
        slice_only = False
        t0 = theta_0[key]

    result = _merge_methods.dispatch(
        calcmode, mode, key,
        t0, theta_1[key],
        theta_2[key] if theta_2 is not None else None,
        current_alpha, current_beta,
    )
    if slice_only:
        theta_0[key][:, 0:4, :, :] = result
    else:
        theta_0[key] = result
    del result, t0
```

Keep all `elif calcmode == "trainDifference":`, `elif calcmode == "smoothAdd":`, etc. blocks unchanged.

- [ ] **Step 5: Verify the import works**

```bash
python3 -c "
import sys
sys.path.insert(0, 'scripts/sd_mecha')
sys.path.insert(0, 'scripts')
import tests.conftest  # installs mocks
from mergers.mergers import smerge
print('smerge importable OK')
"
```

Expected: `smerge importable OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/mergers/mergers.py
git commit -m "feat: replace calcmode dispatch with sd-mecha methods.dispatch()"
```

---

## Task 6: Create `scripts/mergers/streamer.py` (TDD)

**Note on memory model:** `safetensors.save_file()` requires a complete dict — it cannot write incrementally. `streamer.py` builds the output dict key-by-key (releasing each result tensor as it goes, so only one merged tensor is alive at a time), then calls `save_file()` once. The memory saving is in the **result dict** (never fully built), not in the file write itself.

**Files:**
- Create: `tests/test_streamer.py`
- Create: `scripts/mergers/streamer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_streamer.py
import pytest
import torch
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from mergers import streamer, methods


def _make_state_dict(n_keys=4, shape=(4, 4), prefix="model.diffusion_model.input_blocks."):
    """Build a minimal fake state dict with 'weight' keys."""
    return {
        f"{prefix}{i}.weight": torch.full(shape, float(i), dtype=torch.float32)
        for i in range(n_keys)
    }


class TestStreamerBasic:
    def test_output_file_is_created(self, tmp_path):
        theta_0 = _make_state_dict()
        theta_1 = _make_state_dict()
        out = str(tmp_path / "out.safetensors")

        streamer.merge_and_save(
            save_path=out,
            theta_0=theta_0,
            theta_1=theta_1,
            theta_2=None,
            calcmode="normal",
            mode="Weight sum",
            base_alpha=0.5,
            base_beta=0.0,
            weights_a=[], weights_b=[],
            isxl=False, isflux=False,
            useblocks=False, usebeta=False,
            deep=[], randomer=None, lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Off", ex_blocks=[], ex_elems=[""],
        )
        assert os.path.isfile(out)

    def test_output_has_same_keys_as_theta_0(self, tmp_path):
        from safetensors.torch import load_file
        theta_0 = _make_state_dict()
        theta_1 = _make_state_dict()
        out = str(tmp_path / "out.safetensors")

        streamer.merge_and_save(
            save_path=out, theta_0=theta_0, theta_1=theta_1, theta_2=None,
            calcmode="normal", mode="Weight sum",  # full UI mode string
            base_alpha=0.5, base_beta=0.0,
            weights_a=[], weights_b=[],
            isxl=False, isflux=False,
            useblocks=False, usebeta=False,
            deep=[], randomer=None, lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Off", ex_blocks=[], ex_elems=[""],
        )
        result = load_file(out)
        assert set(result.keys()) == set(theta_0.keys())

    def test_weight_mode_alpha_half_produces_midpoint(self, tmp_path):
        from safetensors.torch import load_file
        theta_0 = {"model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4)}
        theta_1 = {"model.diffusion_model.input_blocks.0.weight": torch.ones(4, 4) * 2.0}
        out = str(tmp_path / "out.safetensors")

        streamer.merge_and_save(
            save_path=out, theta_0=theta_0, theta_1=theta_1, theta_2=None,
            calcmode="normal", mode="Weight sum",  # full UI mode string
            base_alpha=0.5, base_beta=0.0,
            weights_a=[], weights_b=[],
            isxl=False, isflux=False,
            useblocks=False, usebeta=False,
            deep=[], randomer=None, lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Off", ex_blocks=[], ex_elems=[""],
        )
        result = load_file(out)
        expected = torch.ones(4, 4)
        assert torch.allclose(result["model.diffusion_model.input_blocks.0.weight"].float(),
                              expected, atol=1e-5)

    def test_stage2_keys_copied_from_theta_1(self, tmp_path):
        """Keys in theta_1 but not theta_0 (e.g. text encoder) are copied to output."""
        from safetensors.torch import load_file
        theta_0 = {"model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4)}
        theta_1 = {
            "model.diffusion_model.input_blocks.0.weight": torch.ones(4, 4),
            "model.diffusion_model.text_encoder.weight": torch.ones(4, 4) * 9.0,
        }
        out = str(tmp_path / "out.safetensors")

        streamer.merge_and_save(
            save_path=out, theta_0=theta_0, theta_1=theta_1, theta_2=None,
            calcmode="normal", mode="Weight sum",  # full UI mode string
            base_alpha=0.5, base_beta=0.0,
            weights_a=[], weights_b=[],
            isxl=False, isflux=False,
            useblocks=False, usebeta=False,
            deep=[], randomer=None, lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Off", ex_blocks=[], ex_elems=[""],
        )
        result = load_file(out)
        assert "model.diffusion_model.text_encoder.weight" in result
```

- [ ] **Step 2: Run to confirm failure**

```bash
python3 -m pytest tests/test_streamer.py -v 2>&1 | head -20
```

Expected: `ImportError` — streamer does not exist yet.

- [ ] **Step 3: Create `scripts/mergers/streamer.py`**

```python
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

from scripts.mergers.mergers import (
    resolve_alpha, CHCKPOINT_DICT_SKIP_ON_MERGE,
)
from scripts.mergers import methods as _methods


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
```

- [ ] **Step 4: Run tests**

```bash
python3 -m pytest tests/test_streamer.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add scripts/mergers/streamer.py tests/test_streamer.py
git commit -m "feat: add streaming save path (streamer.py)"
```

---

## Task 7: Route save path in `smerge()` to `streamer.py`

**Files:**
- Modify: `scripts/mergers/mergers.py` (around line 650–701)

- [ ] **Step 1: Add `streamer` import to `mergers.py`**

After the `methods` import added in Task 5:
```python
from scripts.mergers import streamer as _streamer
```

- [ ] **Step 2: Derive save path and route to streamer**

In `smerge()`, locate the end of the preamble — just before the Stage 1/2 comment (line ~413) and after `to_qdtype` calls. The exact location is after line 407 (`to_qdtype(theta_0, theta_2, ...)`), before line 413 (`##### Stage 1/2`).

Add:
```python
    # ---- Streaming save path ----
    _do_save = SAVEMODES[0] in save_sets or SAVEMODES[1] in save_sets
    if _do_save and calcmode in _merge_methods.IN_SCOPE_CALCMODES:
        # Build output path using the same logic as savemodel() in model_util.py
        import os
        from modules import sd_models as _sd_models
        from modules.shared import cmd_opts as _cmd_opts
        _pre = ".fp16" if "fp16" in save_sets else ""
        _ext = ".safetensors" if "safetensors" in save_sets else ".ckpt"
        _fname = custom_name if custom_name and custom_name != "" else ""
        if not _fname:
            _mname = makemodelname(
                ",".join(str(x) for x in weights_a) if useblocks else str(base_alpha),
                ",".join(str(x) for x in weights_b) if useblocks else str(base_beta),
                model_a, model_b, model_c, base_alpha, base_beta,
                useblocks, mode, calcmode,
            )
            _fname = _mname.replace(" ", "").replace(",", "_").replace("(","_").replace(")","_")
        _fname = _fname if _ext in _fname else _fname + _pre + _ext
        if hasattr(_cmd_opts, 'ckpt_dir') and _cmd_opts.ckpt_dir:
            _save_dir = _cmd_opts.ckpt_dir
        elif hasattr(_cmd_opts, 'ckpt_dirs') and _cmd_opts.ckpt_dirs:
            _save_dir = _cmd_opts.ckpt_dirs[0]
        else:
            _save_dir = _sd_models.model_path
        _save_path = os.path.join(_save_dir, _fname)

        _streamer.merge_and_save(
            save_path=_save_path,
            theta_0=theta_0, theta_1=theta_1, theta_2=theta_2,
            calcmode=calcmode, mode=mode,
            base_alpha=base_alpha, base_beta=base_beta,
            weights_a=weights_a if useblocks else [],
            weights_b=weights_b if useblocks and usebeta else [],
            isxl=isxl, isflux=isflux,
            useblocks=useblocks, usebeta=usebeta,
            deep=deep, randomer=randomer, lucks=lucks,
            deepprint=deepprint, esettings=esettings,
            inex=inex, ex_blocks=ex_blocks, ex_elems=ex_elems,
        )
        # Build currentmodel name for return (same as end of smerge)
        _wa_str = ",".join(str(x) for x in weights_a_excluded if x is not None)
        _wb_str = ",".join(str(x) for x in weights_b_excluded if x is not None)
        currentmodel = makemodelname(_wa_str, _wb_str, model_a, model_b, model_c,
                                     base_alpha, base_beta, useblocks, mode, calcmode)
        modelid = rwmergelog(currentmodel, mergedmodel)
        return f"Merged model saved: {_save_path}", currentmodel, modelid, None, metadata
```

Note: returning `None` for `theta_0` when streaming (no in-memory merged dict). `smergegen()` checks the return value — ensure the `model_loader()` call in `smergegen()` handles `theta_0=None` by skipping load (streaming save doesn't load into running model).

- [ ] **Step 3: Guard `savemodel()` and `model_loader()` in `smergegen()` for streaming path**

In `smergegen()` at lines 136–140:
```python
save = True if SAVEMODES[0] in save_sets else False
result = savemodel(theta_0,currentmodel,custom_name,save_sets,metadata) if save else "Merged model loaded:"+currentmodel
model_loader(checkpoint_info, theta_0, metadata, currentmodel)
```

Replace with:
```python
save = True if SAVEMODES[0] in save_sets else False
if theta_0 is None:
    # streaming save already completed inside smerge(); result is the save path message
    pass  # result is already set from smerge() return value
else:
    result = savemodel(theta_0,currentmodel,custom_name,save_sets,metadata) if save else "Merged model loaded:"+currentmodel
    model_loader(checkpoint_info, theta_0, metadata, currentmodel)
```

Note: `result` at line 126 is set from `smerge()`'s first return value. When `theta_0 is None`, `smerge()` already returned `"Merged model saved: {path}"` as the result string — do not overwrite it with `savemodel()`.

- [ ] **Step 4: Verify import works**

```bash
python3 -c "
import sys
sys.path.insert(0, 'scripts/sd_mecha')
sys.path.insert(0, 'scripts')
import tests.conftest
from mergers.mergers import smerge, smergegen
print('routing imports OK')
"
```

Expected: `routing imports OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/mergers/mergers.py
git commit -m "feat: route save-to-disk merges through streaming save path"
```

---

## Task 8: Add new calcmodes to `supermerger.py` UI

**Files:**
- Modify: `scripts/supermerger.py` (line 42)

- [ ] **Step 1: Update `CALCMODES` list**

Open `scripts/supermerger.py`. Line 42 currently reads:
```python
CALCMODES  = ["normal", "cosineA", "cosineB","trainDifference","smoothAdd","smoothAdd MT","extract","tensor","tensor2","self","plus random"]
```

Replace with:
```python
CALCMODES  = ["normal", "cosineA", "cosineB","trainDifference","smoothAdd","smoothAdd MT","extract","tensor","tensor2","self","plus random","slerp","ties_sum","add_ties_with_dare","dropout"]
```

- [ ] **Step 2: Verify the UI can load (import check)**

```bash
python3 -c "
import sys
sys.path.insert(0, 'scripts/sd_mecha')
sys.path.insert(0, 'scripts')
import tests.conftest
import supermerger
print('CALCMODES:', supermerger.CALCMODES[-4:])
"
```

Expected: last 4 entries are the new calcmodes.

- [ ] **Step 3: Commit**

```bash
git add scripts/supermerger.py
git commit -m "feat: add slerp, ties_sum, add_ties_with_dare, dropout to calcmode UI"
```

---

## Task 9: Final verification

- [ ] **Step 1: Run the full test suite**

```bash
python3 -m pytest tests/ -v
```

Expected: all tests pass with no errors or warnings.

- [ ] **Step 2: Verify precosine is fully removed**

```bash
grep -n "precosine\|sim, sims" scripts/mergers/mergers.py
```

Expected: no results — both the call and the variable references are gone.

- [ ] **Step 3: Verify no remaining direct calcmode math**

```bash
grep -n "torch.lerp\|theta_0_a\s*=" scripts/mergers/mergers.py | grep -v "def \|#"
```

Expected: no `theta_0_a` assignments remain in the key loop; only in `tensormerge()` or other helper functions.

- [ ] **Step 4: Verify new calcmodes are guarded from Triple/Twice**

```bash
grep -n "TWO_MODEL_ONLY" scripts/mergers/mergers.py
```

Expected: one line showing the guard block added in Task 5.

- [ ] **Step 5: Commit final state**

```bash
git add -A
git status  # review — should be clean
git commit -m "chore: final cleanup and verification for sd-mecha migration" --allow-empty
```

---

## Appendix: Key Reference Points in Existing Code

| What | File | Line(s) |
|---|---|---|
| `CALCMODES` list | `scripts/supermerger.py` | 42 |
| `smergegen()` — save + load logic | `scripts/mergers/mergers.py` | 113–163 |
| `smerge()` — entry point | `scripts/mergers/mergers.py` | 220 |
| Preamble: model loading, dtype setup | `scripts/mergers/mergers.py` | 330–407 |
| `precosine` pre-pass (to remove) | `scripts/mergers/mergers.py` | 409–411 |
| Key loop start | `scripts/mergers/mergers.py` | 421 |
| Block/alpha resolution (→ `resolve_alpha`) | `scripts/mergers/mergers.py` | 432–481 |
| Calcmode dispatch block (→ `methods.dispatch`) | `scripts/mergers/mergers.py` | 490–551 |
| Stage 2/2 key copy loop | `scripts/mergers/mergers.py` | 622–626 |
| `smerge()` return | `scripts/mergers/mergers.py` | 701 |
| `savemodel()` path derivation | `scripts/mergers/model_util.py` | 96–112 |
| `makemodelname()` | `scripts/mergers/mergers.py` | 1009 |
