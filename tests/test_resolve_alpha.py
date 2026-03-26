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
        # For SD 1.5, blockfromkey returns BLOCKID values.
        # BLOCKID[1] = "IN00" for input_blocks.0.
        # "IN00" is at BLOCKIDXLLL index 3, and BLOCKID index 1.
        # With weight_index=1 (>0), current_alpha = weights_a[weight_index-1] = weights_a[0].
        # Set weights_a[0] = 0.25 (distinct from base_alpha=0.5) to confirm per-block lookup.
        weights_a = [0.25] + [0.5] * 25  # BASE slot = 0.25, rest = 0.5
        alpha, beta, skip = resolve_alpha(
            key="model.diffusion_model.input_blocks.0.0.weight",
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
        # input_blocks.0 → IN00 → BLOCKID index 1; weights_a[index-1] = weights_a[0] = 0.25
        assert skip is False
        assert abs(alpha - 0.25) < 1e-6, (
            f"Expected per-block alpha=0.25, got {alpha}. "
            "Check BLOCKID table to confirm IN00 index."
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
