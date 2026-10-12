"""Unit tests for E049 summarize.py."""
from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path
from summarize import read_batch, group_by, mean_std, wilcoxon_paired, ALL_ARMS, COMPARISONS


class TestMeanStd:
    def test_mean_std(self):
        mu, sd = mean_std([0.7, 0.8, 0.75])
        assert abs(mu - 0.75) < 1e-9
        assert sd > 0

    def test_single_value(self):
        mu, sd = mean_std([0.7])
        assert mu == 0.7
        assert sd == 0.0

    def test_empty(self):
        mu, sd = mean_std([])
        assert mu != mu  # NaN


class TestGroupBy:
    def test_basic_grouping(self):
        rows = [
            {"baseline": "mcat", "arm": "native", "score": 0.7},
            {"baseline": "mcat", "arm": "native", "score": 0.72},
            {"baseline": "mcat", "arm": "learned", "score": 0.75},
        ]
        grouped = group_by(rows, "baseline", "arm")
        assert grouped[("mcat", "native")] == [0.7, 0.72]
        assert grouped[("mcat", "learned")] == [0.75]


class TestWilcoxonPaired:
    def test_identical_returns_one(self):
        p, n = wilcoxon_paired([0.70, 0.72, 0.74, 0.71, 0.73], [0.70, 0.72, 0.74, 0.71, 0.73])
        assert n == 5
        assert p == 1.0

    def test_insufficient_data_returns_nan(self):
        p, n = wilcoxon_paired([0.7], [0.8])
        assert p != p  # NaN

    def test_known_difference(self):
        # If scipy is available, test the sign-test implementation
        try:
            import scipy.stats
            a = [0.70, 0.72, 0.71, 0.73, 0.69]
            b = [0.75, 0.77, 0.76, 0.78, 0.74]
            p, n = wilcoxon_paired(a, b)
            assert n == 5
            assert 0.0 < p <= 1.0  # valid p-value in range
            assert p < 0.1  # directional signal present (all a < b)
        except ImportError:
            pass  # scipy not available, skip


class TestReadBatch:
    def test_reads_completions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Path(tmpdir)
            completion = {
                "status": "complete",
                "task_id": "outer_test/mcat/learned/fold0/seed3",
                "baseline": "mcat",
                "arm": "learned",
                "fold": 0,
                "seed": 3,
                "protocol": "outer_test",
                "score": 0.7238,
            }
            (batch / "protocol=mcat").mkdir()
            p = batch / "protocol=mcat" / "completion.json"
            p.write_text(json.dumps(completion), encoding="utf-8")

            rows = list(read_batch(batch))
            assert len(rows) == 1
            assert rows[0]["score"] == 0.7238

    def test_skips_incomplete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            batch = Path(tmpdir)
            (batch / "task1" / "completion.json").parent.mkdir()
            (batch / "task1" / "completion.json").write_text(
                json.dumps({"status": "failed", "error": "crash"}), encoding="utf-8")
            rows = list(read_batch(batch))
            # Failed tasks are silently skipped; returns empty list for partial batches
            assert rows == []

    def test_empty_batch_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError):
                list(read_batch(Path(tmpdir)))


class TestComparisons:
    def test_all_comparisons_have_valid_arms(self):
        for label, arm_a, arm_b, _ in COMPARISONS:
            assert arm_a in ALL_ARMS, f"{label}: arm_a {arm_a!r} not in ALL_ARMS"
            assert arm_b in ALL_ARMS, f"{label}: arm_b {arm_b!r} not in ALL_ARMS"

    def test_comparison_coverage(self):
        arms_in_comparisons = set()
        for _, a, b, _ in COMPARISONS:
            arms_in_comparisons.add(a)
            arms_in_comparisons.add(b)
        assert "native" in arms_in_comparisons
        assert "learned" in arms_in_comparisons
        assert "independent" in arms_in_comparisons
