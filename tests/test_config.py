"""
Tests for src/config.py

Covers:
- resolve_api_key(): precedence, missing key, SystemExit
- resolve_db_path(): auto-detection, explicit path, path traversal rejection,
  magic bytes validation, missing file
- validate_ulid(): valid ULIDs, invalid formats, injection attempts
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import pytest

from src.config import (
    _SQLITE_MAGIC,
    ENV_ANTHROPIC_KEY,
    ENV_OPENAI_KEY,
    UserConfig,
    load_config,
    resolve_api_key,
    resolve_db_path,
    save_config,
    validate_ulid,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sqlite_file(path: Path) -> None:
    """Write a minimal valid SQLite3 file (magic bytes only)."""
    path.write_bytes(_SQLITE_MAGIC + b"\x00" * 84)  # 100-byte header


def _make_non_sqlite_file(path: Path) -> None:
    """Write a file that is NOT a SQLite database."""
    path.write_bytes(b"PK\x03\x04" + b"\x00" * 96)  # ZIP magic


# ---------------------------------------------------------------------------
# resolve_api_key
# ---------------------------------------------------------------------------


class TestResolveApiKey:
    def test_explicit_flag_takes_precedence_over_env(self):
        with mock.patch.dict(os.environ, {ENV_ANTHROPIC_KEY: "env-key"}):
            result = resolve_api_key("anthropic", explicit="flag-key")
        assert result == "flag-key"

    def test_reads_anthropic_key_from_env(self):
        with mock.patch.dict(os.environ, {ENV_ANTHROPIC_KEY: "sk-ant-test123"}):
            result = resolve_api_key("anthropic")
        assert result == "sk-ant-test123"

    def test_reads_openai_key_from_env(self):
        with mock.patch.dict(os.environ, {ENV_OPENAI_KEY: "sk-openai-test"}):
            result = resolve_api_key("openai")
        assert result == "sk-openai-test"

    def test_strips_whitespace_from_env_value(self):
        with mock.patch.dict(os.environ, {ENV_ANTHROPIC_KEY: "  sk-ant-padded  "}):
            result = resolve_api_key("anthropic")
        assert result == "sk-ant-padded"

    def test_missing_key_exits_with_code_1(self):
        env = {k: v for k, v in os.environ.items() if k not in (ENV_ANTHROPIC_KEY, ENV_OPENAI_KEY)}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit) as exc_info:
                resolve_api_key("anthropic")
        assert exc_info.value.code == 1

    def test_missing_key_message_does_not_contain_key_value(self, capsys):
        """Security: error message must not echo any key material."""
        env = {k: v for k, v in os.environ.items() if k not in (ENV_ANTHROPIC_KEY, ENV_OPENAI_KEY)}
        with mock.patch.dict(os.environ, env, clear=True):
            with pytest.raises(SystemExit):
                resolve_api_key("anthropic")
        captured = capsys.readouterr()
        # The error text should mention the env var name but not any key value
        assert "ANTHROPIC_API_KEY" in captured.err
        assert "sk-" not in captured.err

    def test_empty_string_env_value_triggers_exit(self):
        with mock.patch.dict(os.environ, {ENV_ANTHROPIC_KEY: ""}):
            with pytest.raises(SystemExit) as exc_info:
                resolve_api_key("anthropic")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# resolve_db_path
# ---------------------------------------------------------------------------


class TestResolveDbPath:
    def test_explicit_valid_sqlite_path_is_returned(self, tmp_path):
        db = tmp_path / "opencode.db"
        _make_sqlite_file(db)
        result = resolve_db_path(str(db))
        assert result == str(db)

    def test_path_traversal_is_rejected(self):
        with pytest.raises(ValueError, match="traversal"):
            resolve_db_path("/tmp/../etc/passwd")

    def test_path_traversal_with_tilde_prefix_rejected(self):
        with pytest.raises(ValueError, match="traversal"):
            resolve_db_path("~/../../../etc/shadow")

    def test_non_sqlite_file_is_rejected(self, tmp_path):
        db = tmp_path / "fake.db"
        _make_non_sqlite_file(db)
        with pytest.raises(ValueError, match="SQLite"):
            resolve_db_path(str(db))

    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent.db"
        with pytest.raises(FileNotFoundError):
            resolve_db_path(str(missing))

    def test_missing_explicit_error_mentions_path(self, tmp_path):
        missing = tmp_path / "nonexistent.db"
        with pytest.raises(FileNotFoundError, match="nonexistent.db"):
            resolve_db_path(str(missing))

    def test_no_default_db_raises_file_not_found(self, tmp_path):
        """When auto-detecting and no DB exists, raises FileNotFoundError."""
        with mock.patch("src.config.DEFAULT_DB_PATHS", [str(tmp_path / "nope.db")]):
            with pytest.raises(FileNotFoundError, match="standard location"):
                resolve_db_path()

    def test_auto_detect_finds_first_valid_db(self, tmp_path):
        db1 = tmp_path / "first.db"
        db2 = tmp_path / "second.db"
        _make_sqlite_file(db1)
        _make_sqlite_file(db2)
        with mock.patch("src.config.DEFAULT_DB_PATHS", [str(db1), str(db2)]):
            result = resolve_db_path()
        assert result == str(db1)

    def test_auto_detect_skips_missing_paths(self, tmp_path):
        missing = tmp_path / "missing.db"
        real = tmp_path / "real.db"
        _make_sqlite_file(real)
        with mock.patch("src.config.DEFAULT_DB_PATHS", [str(missing), str(real)]):
            result = resolve_db_path()
        assert result == str(real)

    def test_symlink_path_is_rejected(self, tmp_path):
        """Security: symlinks must be rejected, per the resolve_db_path contract."""
        from src.errors import ConfigurationError

        real = tmp_path / "real.db"
        _make_sqlite_file(real)
        link = tmp_path / "link.db"
        link.symlink_to(real)
        with pytest.raises(ConfigurationError, match="symbolic link"):
            resolve_db_path(str(link))


# ---------------------------------------------------------------------------
# validate_ulid
# ---------------------------------------------------------------------------


class TestValidateUlid:
    VALID_ULID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"

    def test_valid_ulid_is_returned_unchanged(self):
        assert validate_ulid(self.VALID_ULID) == self.VALID_ULID

    def test_lowercase_ulid_is_rejected(self):
        with pytest.raises(ValueError):
            validate_ulid(self.VALID_ULID.lower())

    def test_too_short_is_rejected(self):
        with pytest.raises(ValueError):
            validate_ulid("01ARZ3NDEK")

    def test_too_long_is_rejected(self):
        with pytest.raises(ValueError):
            validate_ulid(self.VALID_ULID + "X")

    def test_sql_injection_attempt_is_rejected(self):
        with pytest.raises(ValueError):
            validate_ulid("'; DROP TABLE session; --")

    def test_empty_string_is_rejected(self):
        with pytest.raises(ValueError):
            validate_ulid("")

    def test_label_appears_in_error_message(self):
        with pytest.raises(ValueError, match="session ID"):
            validate_ulid("bad", label="session ID")


# ---------------------------------------------------------------------------
# Security audit S1: symlink validation in resolve_db_path() must NOT be
# bypassed by relative paths. The audit found that a relative path like
# "link_dir/db.sqlite" can slip past the check if the implementation
# only inspects the leaf path or calls .resolve() too early.
# ---------------------------------------------------------------------------


class TestRelativeSymlinkRejected:
    """A relative path traversing a symlinked directory must be rejected.

    The existing test_symlink_path_is_rejected covers an absolute
    symlink. This test exercises the bypass: a RELATIVE path whose
    FIRST component is a symlinked directory. The anti-symlink check
    must inspect the original path (after expanduser, before resolve)
    and every intermediate component, not just the final path.
    """

    def test_relative_path_through_symlinked_dir_is_rejected(self, tmp_path, monkeypatch) -> None:
        """A relative path like 'link_dir/db.sqlite' must raise ConfigurationError."""
        from src.errors import ConfigurationError

        # Build a real target file and a symlinked directory that
        # points to it. The audit bypass works because the FINAL
        # component (db.sqlite) is a real file, but the FIRST
        # component (link_dir) is a symlink.
        target_dir = tmp_path / "real_dir"
        target_dir.mkdir()
        real_db = target_dir / "db.sqlite"
        _make_sqlite_file(real_db)

        link_dir = tmp_path / "link_dir"
        link_dir.symlink_to(target_dir)

        # chdir into tmp_path so the relative path is interpreted
        # relative to it. monkeypatch.chdir restores the original
        # cwd on teardown, even if the test fails.
        monkeypatch.chdir(tmp_path)

        # The relative path goes THROUGH the symlinked directory.
        # A correct implementation rejects this; the vulnerable one
        # only checks the leaf path and lets it through.
        relative_path = "link_dir/db.sqlite"

        with pytest.raises(ConfigurationError, match="symbolic link"):
            resolve_db_path(relative_path)

    def test_relative_path_with_symlink_in_middle_is_rejected(self, tmp_path, monkeypatch) -> None:
        """A relative path with a symlink in the middle component is rejected."""
        from src.errors import ConfigurationError

        # Build: tmp_path/real/ → tmp_path/link/mid/db.sqlite
        # where `link` is a symlink to `real` and `mid` is a real dir.
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        mid_dir = real_dir / "mid"
        mid_dir.mkdir()
        real_db = mid_dir / "db.sqlite"
        _make_sqlite_file(real_db)

        link_dir = tmp_path / "link"
        link_dir.symlink_to(real_dir)

        monkeypatch.chdir(tmp_path)

        # The path traverses a symlink in the middle, not at the end.
        relative_path = "link/mid/db.sqlite"

        with pytest.raises(ConfigurationError, match="symbolic link"):
            resolve_db_path(relative_path)


# ---------------------------------------------------------------------------
# UserConfig
# ---------------------------------------------------------------------------


class TestUserConfig:
    def test_save_and_load_config_roundtrip(self, tmp_path, monkeypatch):
        """Config is securely saved to disk and can be loaded back."""

        # Mock Path.home() to point to tmp_path
        def mock_home():
            return tmp_path

        monkeypatch.setattr(Path, "home", mock_home)

        config = UserConfig(
            api_key="sk-nim-123",
            base_url="https://integrate.api.nvidia.com/v1",
            model_id="meta/llama-3.1-70b-instruct",
        )

        save_config(config)

        # Check permissions
        config_path = tmp_path / ".config" / "prompt-architect-analyst" / "config.json"
        assert config_path.exists()

        # In Linux, 0o600 is required for security
        mode = config_path.stat().st_mode & 0o777
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"

        # Load it back
        loaded = load_config()
        assert loaded.api_key == "sk-nim-123"
        assert loaded.base_url == "https://integrate.api.nvidia.com/v1"
        assert loaded.model_id == "meta/llama-3.1-70b-instruct"

    def test_load_config_returns_empty_when_missing(self, tmp_path, monkeypatch):
        """Missing config returns empty defaults."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        loaded = load_config()
        assert loaded.api_key == ""
        assert loaded.base_url == ""
        assert loaded.model_id == ""

    def test_load_config_handles_malformed_json(self, tmp_path, monkeypatch):
        """Corrupt config file is gracefully ignored."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        config_dir = tmp_path / ".config" / "prompt-architect-analyst"
        config_dir.mkdir(parents=True)
        config_path = config_dir / "config.json"
        config_path.write_text("{ malformed JSON", encoding="utf-8")

        loaded = load_config()
        assert loaded.api_key == ""
