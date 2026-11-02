#!/usr/bin/env bash
# Download all public datasets for LTN (Tier-1; no IBM account involved).
# Idempotent: curl -C - resumes partial files; existing complete files are skipped.
set -uo pipefail
ROOT=/home/mucbp/PROJECTS/QuTen/data/raw
QT=/home/mucbp/.conda/envs/qt/bin
LOG() { echo "[$(date -u +%H:%M:%S)] $*"; }

fetch() {  # fetch <url> <dest>
  local url=$1 dest=$2
  if [ -s "$dest" ] && [ -z "${3:-}" ]; then LOG "have $(basename "$dest")"; return 0; fi
  for i in 1 2 3; do
    curl -sSL -C - --retry 3 --retry-delay 10 -o "$dest" "$url" && { LOG "ok $(basename "$dest") ($(du -h "$dest" | cut -f1))"; return 0; }
    LOG "retry $i failed for $url"; sleep 20
  done
  LOG "FAILED $url"; return 1
}

# --- DAQEC v1 (Zenodo 18045662, 0.4 MB total) — full record via zenodo_get
LOG "DAQEC v1"
$QT/zenodo_get 18045662 -o "$ROOT/daqec" 2>&1 | tail -2

# --- DAQEC v2 (Zenodo 18087905, 66 KB)
LOG "DAQEC v2"
$QT/zenodo_get 18087905 -o "$ROOT/daqec_v2" 2>&1 | tail -2

# --- SNAP Bitcoin trust graphs (E8)
LOG "SNAP"
fetch "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"   "$ROOT/snap/soc-sign-bitcoinotc.csv.gz"
fetch "https://snap.stanford.edu/data/soc-sign-bitcoinalpha.csv.gz" "$ROOT/snap/soc-sign-bitcoinalpha.csv.gz"

# --- Silicon spin qubits (Zenodo 15653466): selective — skip the 291 MB GST HTML reports
LOG "silicon (selective)"
SIL="https://zenodo.org/records/15653466/files"
fetch "$SIL/analysisFiles.zip?download=1"                       "$ROOT/silicon/analysisFiles.zip"
fetch "$SIL/AncillaryFiles.zip?download=1"                      "$ROOT/silicon/AncillaryFiles.zip"
fetch "$SIL/raw_figure_data.zip?download=1"                     "$ROOT/silicon/raw_figure_data.zip"
fetch "$SIL/GSTdataset_parity_DeviceA_3110.txt?download=1"      "$ROOT/silicon/GSTdataset_parity_DeviceA_3110.txt"
fetch "$SIL/GSTdataset_parity_DeviceB_28574.txt?download=1"     "$ROOT/silicon/GSTdataset_parity_DeviceB_28574.txt"
fetch "https://zenodo.org/records/15653466/files/2Q_GST_Parity_Readout_Demo_Script%201.py?download=1" "$ROOT/silicon/2Q_GST_Parity_Readout_Demo_Script_1.py"

# --- Google QEC (smallest -> largest; 65 GB repetition zip deliberately skipped [DEV-8])
LOG "google sycamore 2022 (0.32 GB)"
fetch "https://zenodo.org/records/6804040/files/google_qec3v5_experiment_data.zip?download=1" "$ROOT/google/google_qec3v5_experiment_data.zip"
LOG "google color code (0.74 GB)"
fetch "https://zenodo.org/records/14238944/files/google_superdense_color_code_d3_d5.zip?download=1" "$ROOT/google/google_superdense_color_code_d3_d5.zip"
LOG "google dynamic circuits (2.08 GB)"
fetch "https://zenodo.org/records/14238907/files/google_dynamic_circuits_d3_d5.zip?download=1" "$ROOT/google/google_dynamic_circuits_d3_d5.zip"
LOG "google willow 105Q (5.72 GB)"
fetch "https://zenodo.org/records/13273331/files/google_105Q_surface_code_d3_d5_d7.zip?download=1" "$ROOT/google/google_105Q_surface_code_d3_d5_d7.zip"

# --- integrity: test all zips
LOG "zip integrity"
for z in "$ROOT"/google/*.zip "$ROOT"/silicon/*.zip; do
  python3 - "$z" <<'EOF' || echo "ZIP-BAD $z"
import sys, zipfile
z = sys.argv[1]
zipfile.ZipFile(z).testzip
print(f"zip-ok {z.split('/')[-1]} entries={len(zipfile.ZipFile(z).namelist())}")
EOF
done
LOG "ALL DONE"
