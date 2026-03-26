# tests/conftest.py
"""
Mock A1111/Forge WebUI modules so tests can import from scripts/ without
needing a running WebUI environment.
"""
import sys
import types
import os

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
    sd_samplers=_make_mock_module("modules.sd_samplers", samplers=[]),
    scripts=_make_mock_module("modules.scripts", basedir=lambda: "/tmp"),
    devices=_make_mock_module("modules.devices", torch_gc=lambda: None),
    extras=_make_mock_module("modules.extras"),
    script_callbacks=_make_mock_module("modules.script_callbacks"),
    extra_networks=_make_mock_module("modules.extra_networks"),
    launch_utils=_make_mock_module("modules.launch_utils", git_tag=lambda: "test"),
)
_modules_stub.ui = _make_mock_module("modules.ui",
    plaintext_to_html=lambda x: x,
    create_output_panel=lambda *a, **kw: None,
    create_refresh_button=lambda *a, **kw: None,
)
_modules_stub.ui_components = _make_mock_module("modules.ui_components",
    ResizeHandleRow=object,
)
_modules_stub.generation_parameters_copypaste = _make_mock_module(
    "modules.generation_parameters_copypaste",
    create_override_settings_dict=lambda *a, **kw: {},
)
# Attributes needed from modules.processing
_modules_stub.processing.create_infotext = lambda *a, **kw: ""
_modules_stub.processing.Processed = object
# Attributes needed from modules.sd_models
_modules_stub.sd_models.unload_model_weights = lambda *a, **kw: None
_modules_stub.sd_models.get_closet_checkpoint_match = lambda *a, **kw: None
# Attributes needed from modules.shared
_modules_stub.shared.opts = object()
_modules_stub.shared.cmd_opts = object()

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
    ("modules.ui_components", _modules_stub.ui_components),
    ("modules.generation_parameters_copypaste", _modules_stub.generation_parameters_copypaste),
    ("modules.extra_networks", _modules_stub.extra_networks),
    ("modules.launch_utils", _modules_stub.launch_utils),
    ("launch", _make_mock_module("launch", git_tag=lambda: "test")),
]:
    sys.modules.setdefault(mod_name, mod)

# backend stubs for Forge
for name in ["backend", "backend.memory_management", "backend.utils"]:
    sys.modules.setdefault(name, _make_mock_module(name,
        load_torch_file=lambda *a, **kw: {}))
