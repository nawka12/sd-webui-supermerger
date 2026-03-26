# tests/test_streamer.py
import pytest
import torch
import tempfile
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from mergers import streamer, methods
from safetensors.torch import save_file as _save_file


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
            calcmode="normal", mode="Weight sum",
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
            calcmode="normal", mode="Weight sum",
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
            calcmode="normal", mode="Weight sum",
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


class TestDetectArch:
    def _write(self, path, sd):
        _save_file(sd, str(path))
        return str(path)

    def test_sd15_model(self, tmp_path):
        p = self._write(tmp_path / "sd15.safetensors", {
            "model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4),
        })
        isxl, isflux, keys, dtype = streamer.detect_arch(p)
        assert not isxl
        assert not isflux
        assert "model.diffusion_model.input_blocks.0.weight" in keys
        assert dtype == torch.float32

    def test_xl_model(self, tmp_path):
        p = self._write(tmp_path / "xl.safetensors", {
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight": torch.zeros(4, 4),
            "model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4),
        })
        isxl, isflux, keys, dtype = streamer.detect_arch(p)
        assert isxl
        assert not isflux

    def test_flux_model(self, tmp_path):
        p = self._write(tmp_path / "flux.safetensors", {
            "double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(4, dtype=torch.bfloat16),
        })
        isxl, isflux, keys, dtype = streamer.detect_arch(p)
        assert not isxl
        assert isflux
        assert dtype == torch.bfloat16
