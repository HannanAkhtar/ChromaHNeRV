import ast
import builtins
import importlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_no_active_pytorchvideo_import_or_primary_requirement():
    for path in ROOT.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            node.module for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        ]
        assert not any(name.startswith("pytorchvideo") for name in imports)
    requirements = (ROOT / "requirements.txt").read_text(
        encoding="utf-8").lower()
    assert "pytorchvideo" not in requirements
    assert "decord" in requirements


def test_model_all_import_does_not_request_pytorchvideo(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("timm")
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.startswith("pytorchvideo"):
            raise AssertionError("model_all requested pytorchvideo")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    importlib.import_module("model_all")
