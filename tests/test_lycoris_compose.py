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


class TestComposeLoKr:
    def test_plain_w1_w2(self):
        w1 = torch.eye(2)
        w2 = torch.eye(3)
        sd = {
            "mod.lokr_w1": w1,
            "mod.lokr_w2": w2,
        }
        delta = compose_delta(sd, "mod")
        assert torch.allclose(delta, torch.kron(w1, w2))

    def test_factorized_w1(self):
        w1a = torch.randn(4, 2)
        w1b = torch.randn(2, 4)
        w2 = torch.eye(2)
        sd = {
            "mod.lokr_w1_a": w1a,
            "mod.lokr_w1_b": w1b,
            "mod.lokr_w2": w2,
        }
        delta = compose_delta(sd, "mod")
        assert torch.allclose(delta, torch.kron(w1a @ w1b, w2), atol=1e-5)

    def test_factorized_w2(self):
        w1 = torch.eye(2)
        w2a = torch.randn(4, 2)
        w2b = torch.randn(2, 4)
        sd = {
            "mod.lokr_w1": w1,
            "mod.lokr_w2_a": w2a,
            "mod.lokr_w2_b": w2b,
        }
        delta = compose_delta(sd, "mod")
        assert torch.allclose(delta, torch.kron(w1, w2a @ w2b), atol=1e-5)

    def test_alpha_scaling_with_factorized_w1(self):
        w1a = torch.eye(2)
        w1b = torch.eye(2)    # lora_dim = w1b.shape[0] = 2
        w2 = torch.eye(2)
        sd = {
            "mod.lokr_w1_a": w1a,
            "mod.lokr_w1_b": w1b,
            "mod.lokr_w2": w2,
            "mod.alpha": torch.tensor(1.0),  # scale = 1/2 = 0.5
        }
        delta = compose_delta(sd, "mod")
        expected = torch.kron(w1a @ w1b, w2) * 0.5
        assert torch.allclose(delta, expected, atol=1e-5)

    def test_nonfinite_alpha_uses_scale_1(self):
        w1a = torch.eye(2)
        w1b = torch.eye(2)
        w2 = torch.eye(2)
        sd = {
            "mod.lokr_w1_a": w1a,
            "mod.lokr_w1_b": w1b,
            "mod.lokr_w2": w2,
            "mod.alpha": torch.tensor(float("inf")),
        }
        delta = compose_delta(sd, "mod")
        # inf alpha → scale=1.0
        assert torch.allclose(delta, torch.kron(w1a @ w1b, w2), atol=1e-5)

    def test_plain_w1_no_alpha_scale_1(self):
        w1 = torch.full((2, 2), 2.0)
        w2 = torch.full((2, 2), 3.0)
        sd = {
            "mod.lokr_w1": w1,
            "mod.lokr_w2": w2,
        }
        delta = compose_delta(sd, "mod")
        # no lora_dim (no factorized), scale=1.0
        assert torch.allclose(delta, torch.kron(w1, w2))


class TestComposeLoHa:
    def test_plain_loha(self):
        w1a = torch.full((3, 2), 1.0)
        w1b = torch.full((2, 3), 1.0)
        w2a = torch.full((3, 2), 2.0)
        w2b = torch.full((2, 3), 2.0)
        sd = {
            "mod.hada_w1_a": w1a,
            "mod.hada_w1_b": w1b,
            "mod.hada_w2_a": w2a,
            "mod.hada_w2_b": w2b,
        }
        delta = compose_delta(sd, "mod")
        expected = (w1a @ w1b) * (w2a @ w2b)
        assert torch.allclose(delta, expected)

    def test_alpha_scaling(self):
        w1a = torch.eye(2)
        w1b = torch.eye(2)    # dim = w1b.shape[0] = 2
        w2a = torch.eye(2)
        w2b = torch.eye(2)
        sd = {
            "mod.hada_w1_a": w1a,
            "mod.hada_w1_b": w1b,
            "mod.hada_w2_a": w2a,
            "mod.hada_w2_b": w2b,
            "mod.alpha": torch.tensor(1.0),  # scale = 1/2 = 0.5
        }
        delta = compose_delta(sd, "mod")
        # (eye * eye) * 0.5
        assert torch.allclose(delta, torch.eye(2) * 0.5)

    def test_loha_detection_wins_over_lora(self):
        """LoHa detection takes priority when both hada_ and lora_ keys present."""
        sd = {
            "mod.hada_w1_a": torch.eye(2),
            "mod.hada_w1_b": torch.eye(2),
            "mod.hada_w2_a": torch.eye(2),
            "mod.hada_w2_b": torch.eye(2),
            "mod.lora_down.weight": torch.full((2, 2), 99.0),
            "mod.lora_up.weight": torch.full((2, 2), 99.0),
        }
        delta = compose_delta(sd, "mod")
        # LoHa wins: (eye @ eye) * (eye @ eye) = eye, scale=1
        assert torch.allclose(delta, torch.eye(2))

    def test_tucker_variant(self):
        # Tucker LoHa: t1, t2 are core tensors
        rank = 2
        out, inp = 3, 3
        t1 = torch.randn(rank, rank)
        w1a = torch.randn(rank, out)
        w1b = torch.randn(rank, inp)
        t2 = torch.randn(rank, rank)
        w2a = torch.randn(rank, out)
        w2b = torch.randn(rank, inp)
        sd = {
            "mod.hada_w1_a": w1a,
            "mod.hada_w1_b": w1b,
            "mod.hada_w2_a": w2a,
            "mod.hada_w2_b": w2b,
            "mod.hada_t1": t1,
            "mod.hada_t2": t2,
        }
        delta = compose_delta(sd, "mod")
        # Manually compute expected Tucker parts
        part1 = torch.einsum("i j, i p, j r -> p r", t1, w1a, w1b)
        part2 = torch.einsum("i j, i p, j r -> p r", t2, w2a, w2b)
        expected = part1 * part2
        assert torch.allclose(delta, expected, atol=1e-5)
