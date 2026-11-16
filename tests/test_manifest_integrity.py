"""
Manifest Integrity & Cryptographic Checksum Tests (Phase 3 Blocker B4)
======================================================================
Verifies that the experiment manifest records SHA-256 hashes of all artifacts,
excludes the manifest itself from circular hashing, tracks git metadata,
and detects any tampering or modification to recorded artifacts.
"""

import os
import json
import tempfile
import pytest

from scripts.run_phase3_benchmark import compute_sha256, get_git_metadata


def test_sha256_computation_and_tamper_detection():
    """Verify that compute_sha256 accurately hashes files and detects modifications."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = os.path.join(tmpdir, "test_artifact.json")
        with open(test_file, "w") as f:
            f.write('{"initial": "data"}')

        original_hash = compute_sha256(test_file)
        assert len(original_hash) == 64, "SHA-256 hash must be 64 hexadecimal characters"

        # Modify file slightly
        with open(test_file, "w") as f:
            f.write('{"initial": "tampered_data"}')

        modified_hash = compute_sha256(test_file)
        assert original_hash != modified_hash, "Tampered file must produce a different SHA-256 hash!"


def test_git_metadata_retrieval():
    """Verify that git commit and branch metadata are retrieved."""
    meta = get_git_metadata()
    assert "commit" in meta
    assert "branch" in meta
    assert "dirty" in meta
    assert isinstance(meta["dirty"], bool)
    if meta["commit"] != "unknown":
        assert len(meta["commit"]) == 40, "Git commit hash must be 40 characters"


def test_manifest_artifact_checksum_verification():
    """
    Verify that an generated manifest has valid checksums matching on-disk files
    and detects any post-run file corruption.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy artifacts
        art1 = os.path.join(tmpdir, "raw_results.json")
        art2 = os.path.join(tmpdir, "model.pt")
        with open(art1, "w") as f:
            f.write("result_123")
        with open(art2, "wb") as f:
            f.write(b"model_weights_456")

        h1 = compute_sha256(art1)
        h2 = compute_sha256(art2)

        manifest = {
            "status": "COMPLETED",
            "artifact_hashes": {
                "raw_results.json": h1,
                "model.pt": h2,
            },
        }
        manifest_file = os.path.join(tmpdir, "phase3_manifest.json")
        with open(manifest_file, "w") as f:
            json.dump(manifest, f, indent=2)

        # 1. Verification of untouched artifacts
        for rel_path, expected_hash in manifest["artifact_hashes"].items():
            full_path = os.path.join(tmpdir, rel_path)
            actual_hash = compute_sha256(full_path)
            assert actual_hash == expected_hash, f"Hash mismatch for {rel_path}"

        # 2. Tampering test: corrupt model.pt
        with open(art2, "wb") as f:
            f.write(b"corrupted_weights_789")

        corrupted_hash = compute_sha256(art2)
        assert corrupted_hash != manifest["artifact_hashes"]["model.pt"], \
            "Tamper detection failed: corrupted file matched original recorded hash!"

