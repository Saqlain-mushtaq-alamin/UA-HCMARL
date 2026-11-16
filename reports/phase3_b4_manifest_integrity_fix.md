# Phase 3 Remediation Report: B4 — Cryptographic Provenance & Manifest Integrity

## 1. Original Defect Summary
In `reports/phase3_independent_blockers.md` (Defect B4), the independent audit established that:
- Phase 3 runs produced output JSON/PNG files without cryptographic provenance.
- The manifest lacked Git commit SHA, git dirty status, configuration hash, model weight hashes, and output artifact hashes.
- Detached SHA-256 checksums (`phase3_checksums.sha256`) were missing, precluding automated tamper detection and scientific auditability.

---

## 2. Exact Source Files & Functions Changed
1. **`scripts/run_phase3_benchmark.py`**:
   - `compute_sha256(filepath)`: Computes streaming cryptographic SHA-256 hash using 64KB chunks.
   - `get_git_metadata()`: Queries git via `subprocess` for current commit hash, active branch, dirty state (`git status --porcelain`), and lists modified/untracked files.
   - In `run_phase3_benchmark()` Step 4:
     - Calculates `config_sha256` from canonical JSON serialization of `asdict(cfg)`.
     - Calculates individual SHA-256 checksums for all output artifacts (metrics JSON, statistical summaries, plots, PyTorch model checkpoints, replicate files).
     - Assembles unified `phase3_manifest.json` with timestamp, profile, config, git metadata, and artifact hashes.
     - Emits detached standard format checksum file `phase3_checksums.sha256`:
       ```
       <sha256_hash>  <relative_filename>
       ```

---

## 3. Implementation Details & Manifest Structure
The generated `phase3_manifest.json` adheres to the following specification:
```json
{
  "timestamp": "2026-11-06T02:28:50.110218",
  "profile": "smoke",
  "config": { ... },
  "config_sha256": "0ffbd452a533399934ad90ed56fc20e85c2511abae25742e21e26432858d6056",
  "git": {
    "commit": "5596db073be18b03eae36a7734cf1f3a417cf588",
    "branch": "dev_workplace",
    "dirty": true,
    "modified_files": [ ... ]
  },
  "artifacts": {
    "phase3_raw_results.json": "29b257eeaef60ee2a19fdeb8e136e24480e872b4bbd332a3b9ca1d0d1290f4d9",
    "phase3_statistical_summary.json": "9eaae86731fbdb19661f5bb2fdb8fb3cf510b2886605be5eb258dc6629deb557",
    "fig1_clean_preservation.png": "0167e8b65bb0210c089df2199d2798464e1e620ac06d10d57d9e65989476f7a6",
    ...
  }
}
```

Detached verification is fully compatible with standard GNU / Unix tools (`sha256sum -c phase3_checksums.sha256`) and PowerShell / Python verification scripts.

---

## 4. Tests Added & Verification
- **Test File**: `tests/test_manifest_integrity.py`
  - `test_sha256_computation_and_tamper_detection`: Verifies cryptographic hash accuracy and ensures modifying even 1 byte triggers tamper detection failure.
  - `test_git_metadata_retrieval`: Asserts Git commit SHA is a valid 40-character hexadecimal string, branch is populated, and dirty state is boolean.
  - `test_manifest_artifact_checksum_verification`: Validates `phase3_manifest.json` against actual files in a generated benchmark run and verifies that all artifact hashes in `phase3_checksums.sha256` match reality.

### Verification Command & Output
```bash
python -m pytest tests/test_manifest_integrity.py -v
```
Output:
```
tests/test_manifest_integrity.py::test_sha256_computation_and_tamper_detection PASSED
tests/test_manifest_integrity.py::test_git_metadata_retrieval PASSED
tests/test_manifest_integrity.py::test_manifest_artifact_checksum_verification PASSED
============================== 3 passed in 0.48s ===============================
```

---

## 5. Remaining Limitations
None. Manifest and checksums are automatically generated on every benchmark run.

---

## 6. Evidence Paths
- `scripts/run_phase3_benchmark.py` (lines 48–110, 600–665)
- `tests/test_manifest_integrity.py` (lines 1–70)
- `experiments/phase3_smoke_b1b4/phase3_manifest.json`
- `experiments/phase3_smoke_b1b4/phase3_checksums.sha256`

