"""
Tests for src/utils.py

Covers:
- scrub_paths(): Linux, macOS, Windows paths; tilde paths; safe strings
- ts_from_ms(): valid timestamps, out-of-range values
- safe_json_loads(): valid JSON, invalid JSON, non-dict JSON, empty string
- clamp(): boundaries and interior values
- squash(): saturation at target, beyond target, zero target
- shrink(): zero n, full n, mid n
- run_fingerprint(): determinism, sensitivity to changes, SHA-256 length
- project_label(): Linux, macOS, short paths, empty string
- shannon_evenness(): single tool, equal tools, skewed distribution
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC

import pytest

from src.utils import (
    clamp,
    project_label,
    run_fingerprint,
    safe_json_loads,
    scrub_paths,
    shannon_evenness,
    shrink,
    squash,
    ts_from_ms,
)

# ---------------------------------------------------------------------------
# scrub_paths
# ---------------------------------------------------------------------------


class TestScrubPaths:
    def test_linux_home_path_is_scrubbed(self):
        result = scrub_paths("/home/carlos/projects/app")
        assert "carlos" not in result
        assert "~" in result

    def test_macos_users_path_is_scrubbed(self):
        result = scrub_paths("/Users/alice/Documents/project")
        assert "alice" not in result
        assert "~" in result

    def test_windows_users_path_is_scrubbed(self):
        result = scrub_paths(r"C:\Users\bob\AppData\project")
        assert "bob" not in result

    def test_tilde_path_passes_unchanged(self):
        path = "~/.local/share/opencode/opencode.db"
        assert scrub_paths(path) == path

    def test_relative_path_passes_unchanged(self):
        path = "src/reader/opencode.py"
        assert scrub_paths(path) == path

    def test_safe_string_passes_unchanged(self):
        text = "Processing session 01ARZ3NDEKTSV4RRFFQ69G5FAV across 3 projects"
        assert scrub_paths(text) == text

    def test_multiple_paths_in_one_string(self):
        text = "/home/alice/a and /Users/bob/b"
        result = scrub_paths(text)
        assert "alice" not in result
        assert "bob" not in result


# ---------------------------------------------------------------------------
# ts_from_ms
# ---------------------------------------------------------------------------


class TestTsFromMs:
    VALID_TS = 1_700_000_000_000  # 2023-11-14 in ms

    def test_valid_timestamp_returns_utc_datetime(self):
        dt = ts_from_ms(self.VALID_TS)
        assert dt.tzinfo == UTC

    def test_timestamp_value_is_correct(self):
        dt = ts_from_ms(1_700_000_000_000)
        assert dt.year == 2023

    def test_zero_timestamp_raises_value_error(self):
        with pytest.raises(ValueError, match="range"):
            ts_from_ms(0)

    def test_negative_timestamp_raises_value_error(self):
        with pytest.raises(ValueError):
            ts_from_ms(-1)

    def test_far_future_timestamp_raises_value_error(self):
        with pytest.raises(ValueError):
            ts_from_ms(9_999_999_999_999)

    def test_float_input_is_accepted(self):
        dt = ts_from_ms(float(self.VALID_TS))
        assert dt.year == 2023


# ---------------------------------------------------------------------------
# safe_json_loads
# ---------------------------------------------------------------------------


class TestSafeJsonLoads:
    def test_valid_json_object_is_parsed(self):
        result = safe_json_loads('{"role": "user", "cost": 0.01}')
        assert result == {"role": "user", "cost": 0.01}

    def test_invalid_json_returns_empty_dict(self):
        result = safe_json_loads("{not valid json", context="msg abc")
        assert result == {}

    def test_json_array_returns_empty_dict(self):
        """Top-level arrays are not expected — must return {}."""
        result = safe_json_loads("[1, 2, 3]", context="msg abc")
        assert result == {}

    def test_empty_string_returns_empty_dict(self):
        result = safe_json_loads("", context="msg abc")
        assert result == {}

    def test_invalid_json_does_not_log_content(self, caplog):
        """Security: the raw content of failed JSON must never appear in logs."""
        secret_content = "SECRET_PROMPT_CONTENT_XYZ"
        with caplog.at_level(logging.WARNING, logger="src.utils"):
            safe_json_loads(f'{{"bad": {secret_content}}}', context="session-123")
        assert secret_content not in caplog.text

    def test_context_label_appears_in_log_warning(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.utils"):
            safe_json_loads("{bad", context="message-abc")
        assert "message-abc" in caplog.text


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_value_within_range_unchanged(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_value_below_lo_clamped_to_lo(self):
        assert clamp(-0.1, 0.0, 1.0) == 0.0

    def test_value_above_hi_clamped_to_hi(self):
        assert clamp(1.5, 0.0, 1.0) == 1.0

    def test_exact_lo_is_returned(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0

    def test_exact_hi_is_returned(self):
        assert clamp(1.0, 0.0, 1.0) == 1.0


# ---------------------------------------------------------------------------
# squash
# ---------------------------------------------------------------------------


class TestSquash:
    def test_below_target_returns_fraction(self):
        assert squash(0.5, 1.0) == pytest.approx(0.5)

    def test_at_target_returns_one(self):
        assert squash(1.0, 1.0) == pytest.approx(1.0)

    def test_above_target_saturates_at_one(self):
        assert squash(2.0, 1.0) == pytest.approx(1.0)

    def test_zero_target_returns_one(self):
        """Zero or negative target → signal is maxed (fully saturated)."""
        assert squash(0.0, 0.0) == 1.0

    def test_zero_input_zero_target_returns_one(self):
        assert squash(0.0, 0.0) == 1.0


# ---------------------------------------------------------------------------
# shrink
# ---------------------------------------------------------------------------


class TestShrink:
    def test_zero_n_returns_half(self):
        assert shrink(1.0, 0, 10) == pytest.approx(0.5)

    def test_full_n_returns_raw(self):
        assert shrink(0.8, 10, 10) == pytest.approx(0.8)

    def test_half_n_interpolates(self):
        assert shrink(1.0, 5, 10) == pytest.approx(0.75)

    def test_score_above_half_shrinks_toward_half(self):
        result = shrink(0.9, 3, 10)
        assert 0.5 < result < 0.9

    def test_score_below_half_shrinks_toward_half(self):
        result = shrink(0.2, 3, 10)
        assert 0.2 < result < 0.5

    def test_zero_full_n_returns_raw_unchanged(self):
        """Edge case: full_n == 0 must not divide by zero."""
        assert shrink(0.7, 5, 0) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# run_fingerprint
# ---------------------------------------------------------------------------


class TestRunFingerprint:
    _SAMPLE = [{"text": "hello", "project": "proj", "session": "s1", "idx": 0}]

    def test_returns_16_hex_chars(self):
        fp = run_fingerprint(self._SAMPLE)
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)

    def test_same_input_same_fingerprint(self):
        assert run_fingerprint(self._SAMPLE) == run_fingerprint(self._SAMPLE)

    def test_different_input_different_fingerprint(self):
        other = [{"text": "world", "project": "proj", "session": "s1", "idx": 0}]
        assert run_fingerprint(self._SAMPLE) != run_fingerprint(other)

    def test_empty_list_returns_valid_fingerprint(self):
        fp = run_fingerprint([])
        assert len(fp) == 16

    def test_fingerprint_is_sha256_prefix(self):
        """Security: verify SHA-256 is actually used (not MD5 or SHA-1)."""
        payload = json.dumps(self._SAMPLE, sort_keys=True, ensure_ascii=True)
        expected = hashlib.sha256(payload.encode()).hexdigest()[:16]
        assert run_fingerprint(self._SAMPLE) == expected


# ---------------------------------------------------------------------------
# project_label
# ---------------------------------------------------------------------------


class TestProjectLabel:
    def test_linux_path_removes_username(self):
        result = project_label("/home/carlos/work/myapp")
        assert "carlos" not in result
        assert "myapp" in result

    def test_macos_path_removes_username(self):
        result = project_label("/Users/alice/projects/backend")
        assert "alice" not in result
        assert "backend" in result

    def test_returns_last_two_segments(self):
        result = project_label("/srv/apps/backend")
        assert result == "apps/backend"

    def test_short_path_returns_single_segment(self):
        result = project_label("/myproject")
        assert result == "myproject"

    def test_empty_string_returns_unknown(self):
        assert project_label("") == "unknown"

    def test_windows_style_path(self):
        result = project_label(r"C:\Users\bob\projects\app")
        assert "bob" not in result


# ---------------------------------------------------------------------------
# shannon_evenness
# ---------------------------------------------------------------------------


class TestShannonEvenness:
    def test_single_tool_returns_zero(self):
        assert shannon_evenness({"bash": 10}) == pytest.approx(0.0)

    def test_two_equal_tools_returns_one(self):
        result = shannon_evenness({"bash": 5, "read": 5})
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_skewed_distribution_is_between_zero_and_one(self):
        result = shannon_evenness({"bash": 100, "read": 1, "edit": 1})
        assert 0.0 < result < 1.0

    def test_empty_dict_returns_zero(self):
        assert shannon_evenness({}) == pytest.approx(0.0)

    def test_zero_counts_excluded(self):
        """Tools with zero usage should not count as distinct tools."""
        result_with = shannon_evenness({"bash": 5, "unused": 0})
        result_without = shannon_evenness({"bash": 5})
        assert result_with == pytest.approx(result_without)
