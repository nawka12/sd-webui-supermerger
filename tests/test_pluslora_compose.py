# tests/test_pluslora_compose.py
"""
Integration tests for merge_lora_models_dim with LoKr and LoHa inputs.
Writes synthetic LoRA files to tmp_path, calls merge_lora_models_dim,
and verifies the output contains standard lora_up/lora_down/alpha keys.
"""
import pytest
import torch
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from safetensors.torch import save_file as _save

from mergers.pluslora import merge_lora_models_dim


def _write_lora_lokr(path, rank=4):
    """Write a minimal LoKr safetensors file with one module."""
    w1 = torch.randn(rank, rank)
    w2 = torch.randn(rank, rank)
    sd = {
        "lora_unet_down_blocks_0_attentions_0_to_q.lokr_w1": w1,
        "lora_unet_down_blocks_0_attentions_0_to_q.lokr_w2": w2,
    }
    _save(sd, str(path))
    return str(path)


def _write_lora_loha(path, rank=4):
    """Write a minimal LoHa safetensors file with one module."""
    w1a = torch.randn(rank, rank)
    w1b = torch.randn(rank, rank)
    w2a = torch.randn(rank, rank)
    w2b = torch.randn(rank, rank)
    sd = {
        "lora_unet_down_blocks_0_attentions_0_to_q.hada_w1_a": w1a,
        "lora_unet_down_blocks_0_attentions_0_to_q.hada_w1_b": w1b,
        "lora_unet_down_blocks_0_attentions_0_to_q.hada_w2_a": w2a,
        "lora_unet_down_blocks_0_attentions_0_to_q.hada_w2_b": w2b,
        "lora_unet_down_blocks_0_attentions_0_to_q.alpha": torch.tensor(float(rank)),
    }
    _save(sd, str(path))
    return str(path)


class TestMergeLoraDimLoKr:
    def test_lokr_module_appears_in_output(self, tmp_path):
        f1 = _write_lora_lokr(tmp_path / "lokr1.safetensors")
        f2 = _write_lora_lokr(tmp_path / "lokr2.safetensors")
        ratios = [[1.0] * 26, [1.0] * 26]
        result = merge_lora_models_dim([f1, f2], ratios, new_rank=4,
                                       sets=[], device="cpu",
                                       calc_precision="float")
        mod = "lora_unet_down_blocks_0_attentions_0_to_q"
        assert mod + ".lora_up.weight" in result
        assert mod + ".lora_down.weight" in result
        assert mod + ".alpha" in result

    def test_lokr_output_is_standard_lora_format(self, tmp_path):
        f1 = _write_lora_lokr(tmp_path / "lokr1.safetensors", rank=4)
        ratios = [[1.0] * 26]
        result = merge_lora_models_dim([f1], ratios, new_rank=2,
                                       sets=[], device="cpu",
                                       calc_precision="float")
        mod = "lora_unet_down_blocks_0_attentions_0_to_q"
        assert result[mod + ".lora_down.weight"].shape[0] == 2  # new_rank
        assert float(result[mod + ".alpha"]) == 2.0             # new_rank


class TestMergeLoraDimLoHa:
    def test_loha_module_appears_in_output(self, tmp_path):
        f1 = _write_lora_loha(tmp_path / "loha1.safetensors")
        f2 = _write_lora_loha(tmp_path / "loha2.safetensors")
        ratios = [[1.0] * 26, [1.0] * 26]
        result = merge_lora_models_dim([f1, f2], ratios, new_rank=4,
                                       sets=[], device="cpu",
                                       calc_precision="float")
        mod = "lora_unet_down_blocks_0_attentions_0_to_q"
        assert mod + ".lora_up.weight" in result
        assert mod + ".lora_down.weight" in result

    def test_loha_output_is_standard_lora_format(self, tmp_path):
        f1 = _write_lora_loha(tmp_path / "loha1.safetensors", rank=4)
        ratios = [[1.0] * 26]
        result = merge_lora_models_dim([f1], ratios, new_rank=2,
                                       sets=[], device="cpu",
                                       calc_precision="float")
        mod = "lora_unet_down_blocks_0_attentions_0_to_q"
        assert result[mod + ".lora_down.weight"].shape[0] == 2
        assert float(result[mod + ".alpha"]) == 2.0
