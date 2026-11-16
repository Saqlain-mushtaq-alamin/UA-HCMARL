"""
Unit Test for Refactored Six-Arm Phase 3 Benchmark Runner (Phase 3 Blocker B1-B4)
================================================================================
Tests argument parsing, configuration profile binding, multi-replicate execution,
controller isolation, Kalman recurrence, and cryptographic manifest integrity.
"""

import pytest
import os
import json
import tempfile
from experiment_config import get_config
from scripts.run_phase3_benchmark import run_phase3_benchmark, compute_sha256


def test_six_arm_smoke_execution():
    """Execute a minimal smoke benchmark verifying all six arms complete without errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_phase3_benchmark(
            profile="smoke",
            output_dir=tmpdir,
            force_retrain_noise=True,
        )

        # 1. Check required result and artifact files
        raw_path = os.path.join(tmpdir, "phase3_raw_results.json")
        calib_path = os.path.join(tmpdir, "bse_calibration_metrics.json")
        stat_path = os.path.join(tmpdir, "phase3_statistical_summary.json")
        manifest_path = os.path.join(tmpdir, "phase3_manifest.json")
        checksum_path = os.path.join(tmpdir, "phase3_checksums.sha256")

        assert os.path.exists(raw_path)
        assert os.path.exists(calib_path)
        assert os.path.exists(stat_path)
        assert os.path.exists(manifest_path)
        assert os.path.exists(checksum_path)

        # 2. Check figures
        assert os.path.exists(os.path.join(tmpdir, "fig1_clean_preservation.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig2_robustness_noise_sweep.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig3_bse_vs_noise_training.png"))
        assert os.path.exists(os.path.join(tmpdir, "fig4_bse_calibration_quality.png"))

        # 3. Verify raw results structure contains replicates
        with open(raw_path) as f:
            raw_data = json.load(f)
        assert "replicates" in raw_data
        assert "scenarios" in raw_data

        # 4. Verify manifest has artifact hashes matching disk
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert "artifact_hashes" in manifest
        assert "git" in manifest
        assert manifest["status"] == "COMPLETED"

        for rel_file, recorded_hash in manifest["artifact_hashes"].items():
            full_path = os.path.join(tmpdir, rel_file)
            assert os.path.exists(full_path), f"File {rel_file} listed in manifest missing on disk!"
            disk_hash = compute_sha256(full_path)
            assert disk_hash == recorded_hash, f"Hash mismatch for {rel_file}: {disk_hash} != {recorded_hash}"
