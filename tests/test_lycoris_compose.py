# tests/test_lycoris_compose.py
import pytest
import torch
from mergers.lycoris_compose import compose_delta


class TestComposeLora:
    def test_plain_linear(self):
        up = torch.full((4, 2), 1.0)
        down = torch.full((2, 3), 1.0)
        sd = {
            "mod.lora_up.weight": up,
            "mod.lora_down.weight": down,
        }
        delta = compose_delta(sd, "mod")
        # no alpha → scale=1; up@down = [[2,2,2],...] shape (4,3)
        assert delta.shape == (4, 3)
        assert torch.allclose(delta, up @ down)

    def test_alpha_scaling(self):
        up = torch.eye(2)
        down = torch.eye(2)
        sd = {
            "mod.lora_up.weight": up,
            "mod.lora_down.weight": down,
            "mod.alpha": torch.tensor(1.0),
        }
        delta = compose_delta(sd, "mod")
        # scale = 1 / down.shape[0] = 1/2 = 0.5
        assert torch.allclose(delta, torch.eye(2) * 0.5)

    def test_no_alpha_defaults_scale_1(self):
        up = torch.eye(3)
        down = torch.eye(3)
        sd = {
            "mod.lora_up.weight": up,
            "mod.lora_down.weight": down,
        }
        delta = compose_delta(sd, "mod")
        assert torch.allclose(delta, torch.eye(3))

    def test_conv2d_returns_4d(self):
        # LoCon: down is (rank, in, kH, kW); up is (out, rank, 1, 1)
        up = torch.randn(4, 2, 1, 1)
        down = torch.randn(2, 3, 3, 3)
        sd = {
            "mod.lora_up.weight": up,
            "mod.lora_down.weight": down,
        }
        delta = compose_delta(sd, "mod")
        assert delta.shape == (4, 3, 3, 3)

    def test_conv2d_math_matches_matrix_multiply(self):
        up = torch.randn(4, 2, 1, 1)
        down = torch.randn(2, 3, 1, 1)
        sd = {
            "mod.lora_up.weight": up,
            "mod.lora_down.weight": down,
        }
        delta = compose_delta(sd, "mod")
        expected = (
            up.view(4, 2) @ down.view(2, 3)
        ).view(4, 3, 1, 1)
        assert torch.allclose(delta, expected, atol=1e-5)

    def test_unknown_keys_returns_none(self):
        sd = {"mod.some_weight": torch.randn(4, 4)}
        assert compose_delta(sd, "mod") is None

    def test_target_shape_applied(self):
        up = torch.eye(4)
        down = torch.eye(4)
        sd = {
            "mod.lora_up.weight": up,
            "mod.lora_down.weight": down,
        }
        delta = compose_delta(sd, "mod", target_shape=(2, 2, 2, 2))
        assert delta.shape == (2, 2, 2, 2)

    def test_tucker_mid(self):
        # Tucker mid variant: mid=(2,2), up=(4,2), down=(2,3) → output (4,3)
        rank = 2
        mid = torch.eye(rank)           # (2,2) core
        up  = torch.eye(4)[:, :rank]    # (4,2)
        down = torch.eye(rank, 3)       # (2,3)
        sd = {
            "mod.lora_up.weight": up,
            "mod.lora_down.weight": down,
            "mod.lora_mid.weight": mid,
        }
        delta = compose_delta(sd, "mod")
        # wa = up.T = (2,4), wb = down = (2,3)
        # einsum("ij,ip,jr->pr", eye(2), (2,4), (2,3)) = (4,3)
        wa = up.t()   # (2,4)
        wb = down     # (2,3)
        expected = torch.einsum("i j, i p, j r -> p r", mid, wa, wb)
        assert torch.allclose(delta, expected, atol=1e-5)
