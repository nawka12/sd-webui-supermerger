# tests/test_streamer.py
import pytest
import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from safetensors.torch import save_file as _save_file, load_file as _load_file
from mergers import streamer, methods


def _write_st(path, sd):
    _save_file(sd, str(path))
    return str(path)


class TestMergeAndSave:
    def _simple_sd(self, val=0.0, dtype=torch.float32):
        return {
            "model.diffusion_model.input_blocks.0.weight": torch.full((4, 4), val, dtype=dtype),
            "model.diffusion_model.input_blocks.1.weight": torch.full((4, 4), val + 1.0, dtype=dtype),
        }

    def _call(self, tmp_path, *, path_a, path_b, path_c=None, add_mode=False, alpha=0.5, **kw):
        out = str(tmp_path / "out.safetensors")
        defaults = dict(
            isxl=False, isflux=False,
            useblocks=False, usebeta=False,
        )
        defaults.update(kw)
        streamer.merge_and_save(
            path_a=path_a, path_b=path_b, path_c=path_c,
            save_path=out,
            calcmode="normal", mode="Weight sum",
            base_alpha=alpha, base_beta=0.0,
            weights_a=[], weights_b=[],
            deep=[], randomer=None, lucks={"ceed": 0},
            deepprint=False, esettings=[],
            inex="Off", ex_blocks=[], ex_elems=[""],
            add_mode=add_mode,
            device="cpu",
            **defaults,
        )
        return out

    def test_output_file_is_created(self, tmp_path):
        pa = _write_st(tmp_path / "a.safetensors", self._simple_sd(0.0))
        pb = _write_st(tmp_path / "b.safetensors", self._simple_sd(2.0))
        out = self._call(tmp_path, path_a=pa, path_b=pb)
        assert os.path.isfile(out)

    def test_output_has_correct_keys(self, tmp_path):
        pa = _write_st(tmp_path / "a.safetensors", self._simple_sd(0.0))
        pb = _write_st(tmp_path / "b.safetensors", self._simple_sd(2.0))
        out = self._call(tmp_path, path_a=pa, path_b=pb)
        result = _load_file(out)
        assert set(result.keys()) == {"model.diffusion_model.input_blocks.0.weight",
                                      "model.diffusion_model.input_blocks.1.weight"}

    def test_alpha_half_produces_midpoint(self, tmp_path):
        sd_a = {"model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4)}
        sd_b = {"model.diffusion_model.input_blocks.0.weight": torch.full((4, 4), 2.0)}
        pa = _write_st(tmp_path / "a.safetensors", sd_a)
        pb = _write_st(tmp_path / "b.safetensors", sd_b)
        out = self._call(tmp_path, path_a=pa, path_b=pb, alpha=0.5)
        result = _load_file(out)
        assert torch.allclose(result["model.diffusion_model.input_blocks.0.weight"].float(),
                               torch.ones(4, 4), atol=1e-5)

    def test_stage2_keys_copied_from_b(self, tmp_path):
        """Keys in B not in A (e.g. text encoder) are copied to output."""
        sd_a = {"model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4)}
        sd_b = {
            "model.diffusion_model.input_blocks.0.weight": torch.ones(4, 4),
            "model.diffusion_model.text_encoder.weight": torch.full((4, 4), 9.0),
        }
        pa = _write_st(tmp_path / "a.safetensors", sd_a)
        pb = _write_st(tmp_path / "b.safetensors", sd_b)
        out = self._call(tmp_path, path_a=pa, path_b=pb)
        result = _load_file(out)
        assert "model.diffusion_model.text_encoder.weight" in result

    def test_add_mode_subtracts_c_from_b_inline(self, tmp_path):
        """Add difference: output = A + alpha*(B - C). With A=0, B=3, C=1 → delta=2, result=alpha*2."""
        sd_a = {"model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4)}
        sd_b = {"model.diffusion_model.input_blocks.0.weight": torch.full((4, 4), 3.0)}
        sd_c = {"model.diffusion_model.input_blocks.0.weight": torch.full((4, 4), 1.0)}
        pa = _write_st(tmp_path / "a.safetensors", sd_a)
        pb = _write_st(tmp_path / "b.safetensors", sd_b)
        pc = _write_st(tmp_path / "c.safetensors", sd_c)
        # Add difference mode: delta = B - C = 2, weighted_sum(A=0, delta=2, alpha=0.5) = 1.0
        out = self._call(tmp_path, path_a=pa, path_b=pb, path_c=pc, add_mode=True, alpha=0.5)
        result = _load_file(out)
        assert torch.allclose(result["model.diffusion_model.input_blocks.0.weight"].float(),
                               torch.ones(4, 4), atol=1e-5)

    def test_flux_bare_keys_get_prefix(self, tmp_path):
        """Flux models have bare keys in file; output should have prefixed internal keys."""
        sd_a = {"double_blocks.0.weight": torch.zeros(4, 4)}
        sd_b = {"double_blocks.0.weight": torch.ones(4, 4) * 2.0}
        pa = _write_st(tmp_path / "a.safetensors", sd_a)
        pb = _write_st(tmp_path / "b.safetensors", sd_b)
        out = self._call(tmp_path, path_a=pa, path_b=pb,
                         isxl=False, isflux=True, alpha=0.5)
        result = _load_file(out)
        assert len(result) == 1
        # Flux bare keys get prefixed in output
        assert "model.diffusion_model.double_blocks.0.weight" in result

    def test_dtype_preserved_float16(self, tmp_path):
        """Output tensor should preserve input dtype (float16), not be promoted to float32."""
        sd_a = {"model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4, dtype=torch.float16)}
        sd_b = {"model.diffusion_model.input_blocks.0.weight": torch.full((4, 4), 2.0, dtype=torch.float16)}
        pa = _write_st(tmp_path / "a.safetensors", sd_a)
        pb = _write_st(tmp_path / "b.safetensors", sd_b)
        out = self._call(tmp_path, path_a=pa, path_b=pb, alpha=0.5)
        result = _load_file(out)
        assert result["model.diffusion_model.input_blocks.0.weight"].dtype == torch.float16


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
        assert "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight" in keys
        assert dtype == torch.float32

    def test_flux_model(self, tmp_path):
        p = self._write(tmp_path / "flux.safetensors", {
            "double_blocks.0.img_attn.norm.key_norm.scale": torch.zeros(4, dtype=torch.bfloat16),
        })
        isxl, isflux, keys, dtype = streamer.detect_arch(p)
        assert not isxl
        assert isflux
        assert dtype == torch.bfloat16


class TestDetectArchXlWeights:
    def test_xl_flag_correct_for_xl_file(self, tmp_path):
        """detect_arch returns isxl=True for a file with the SDXL conditioner key."""
        from safetensors.torch import save_file as _sf
        sd = {
            "conditioner.embedders.1.model.transformer.resblocks.9.mlp.c_proj.weight": torch.zeros(4, 4),
            "model.diffusion_model.input_blocks.0.weight": torch.zeros(4, 4),
        }
        path = str(tmp_path / "xl.safetensors")
        _sf(sd, path)
        isxl, isflux, _, _ = streamer.detect_arch(path)
        assert isxl
        assert not isflux
