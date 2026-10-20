"""
Tests for config.resolve_data_dir

Tests:
- environment variable takes priority
- defaults to ./data_store in the current directory
"""

from pathlib import Path

from concordiahub import config


def test_env_var_used(tmp_path, monkeypatch):
    """Test env var overrides the default."""
    monkeypatch.setenv(config.DATA_DIR_ENV_VAR, str(tmp_path / "elsewhere"))
    assert config.resolve_data_dir() == str(tmp_path / "elsewhere")


def test_default_is_cwd_data_store(tmp_path, monkeypatch):
    """Test default is data_store in the current directory."""
    monkeypatch.delenv(config.DATA_DIR_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    assert config.resolve_data_dir() == str(Path(tmp_path) / "data_store")
