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
