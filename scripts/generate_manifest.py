"""
Generate a pipeline manifest recording the state of all inputs and outputs.

Run after `make all` to create a reproducibility snapshot. The manifest
records checksums, timestamps, and verification status so that future
auditors can confirm the pipeline was run cleanly.

Usage: python scripts/generate_manifest.py
"""

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def sha256(filepath):
    """Compute SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def git_hash():
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=ROOT,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "git not found"

def file_info(path):
    """Get file size and modification time."""
    if path.exists():
        stat = path.stat()
        return {
            "exists": True,
            "size_bytes": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "checksum": sha256(path),
        }
    return {"exists": False}


# ─── Build manifest ────────────────────────────────────────────────────────

manifest = {
    "generated": datetime.now().isoformat(),
    "git_commit": git_hash(),
    "python_version": None,
    "raw_data": {},
    "pipeline_outputs": {},
    "figures": {},
    "verification": {},
}

# Python version
import sys
manifest["python_version"] = sys.version

# Raw data checksums
for csv_name in [
    "nsw_credit_transactions_register.csv",
    "nsw_credit_supply_register.csv",
]:
    path = ROOT / "data" / "raw" / "nsw" / csv_name
    manifest["raw_data"][csv_name] = file_info(path)

# Pipeline outputs
for json_name in [
    "abm_results.json",
    "glv_results.json",
    "sensitivity_results.json",
    "demand_robustness_results.json",
    "diagnostics.json",
]:
    path = ROOT / "output" / "results" / json_name
    info = file_info(path)
    manifest["pipeline_outputs"][json_name] = info

# Figures
figure_map = {
    "fig1_problem.pdf": "output/figures/main",
    "fig2_mechanism.pdf": "output/figures/main",
    "fig3_policy.pdf": "output/figures/main",
    "suppfig1_robustness.pdf": "output/figures/supplementary",
}
for fname, fdir in figure_map.items():
    path = ROOT / fdir / fname
    info = file_info(path)
    info["referenced_by"] = "main.tex" if "main" in fdir else "supplementary.tex"
    manifest["figures"][fname] = info

# Run verify_claims and capture result
print("Running verify_claims.py...")
result = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "verify_claims.py")],
    capture_output=True, text=True, cwd=ROOT, timeout=120,
)
all_match = "ALL VALUES MATCH" in result.stdout
model_match = "All model claims match" in result.stdout
manifest["verification"] = {
    "verify_claims_exit_code": result.returncode,
    "empirical_claims_match": all_match,
    "model_claims_match": model_match,
    "mismatches": [],
}
if not all_match:
    for line in result.stdout.split("\n"):
        if "MISMATCH" in line or ": computed=" in line:
            manifest["verification"]["mismatches"].append(line.strip())

# Save
out_path = ROOT / "output" / "MANIFEST.json"
with open(out_path, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"\nManifest saved to {out_path}")
print(f"  Raw data files: {sum(1 for v in manifest['raw_data'].values() if v.get('exists'))}/2")
print(f"  Pipeline outputs: {sum(1 for v in manifest['pipeline_outputs'].values() if v.get('exists'))}/5")
print(f"  Figures: {sum(1 for v in manifest['figures'].values() if v.get('exists'))}/4")
print(f"  Empirical claims match: {all_match}")
print(f"  Model claims match: {model_match}")
