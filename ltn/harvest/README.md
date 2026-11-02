# IBM fleet calibration harvester — token-ready artifact [DEV-1]

**This harvester is never executed against IBM in this project.** The user has no IBM
Quantum account (decision 2026-07-13, `PLAN_DELTA.md` DEV-1), so no live pull has ever
run and none will. It ships as a **complete, dry-run-tested artifact**: the code would
work the day a token exists, and everything short of the authenticated HTTPS exchange
is exercised offline by `--dry-run` and `tests/test_harvester_dryrun.py`.

## One-line setup (the day a token exists)

```bash
/home/mucbp/.conda/envs/qt/bin/python -c "from qiskit_ibm_runtime import QiskitRuntimeService; QiskitRuntimeService.save_account(channel='ibm_quantum_platform', token='<API_TOKEN>', instance='<CRN>', set_as_default=True, overwrite=True)"
```

This writes `~/.qiskit/qiskit-ibm.json`. Environment variables `QISKIT_IBM_TOKEN`,
`QISKIT_IBM_INSTANCE`, `QISKIT_IBM_CHANNEL` take precedence over the file (use them as
CI secrets for the daily cron of plan §2.3 A1). Only channels
`ibm_quantum_platform` (default) and `ibm_cloud` exist in qiskit-ibm-runtime 0.47.0;
the legacy `ibm_quantum` channel was removed. Find your CRN with
`QiskitRuntimeService().instances()`.

## Usage

```bash
PY=/home/mucbp/.conda/envs/qt/bin/python   # the project interpreter, always

# Offline end-to-end rehearsal (no credentials, no network):
$PY -m ltn.harvest.ibm_harvester --out /tmp/ibm_live --dry-run

# Live daily pull (cron this; §2.3 A1):
$PY -m ltn.harvest.ibm_harvester --out data/ibm_live --instance <CRN>

# Historical backfill (§2.3 A5; properties(datetime=...) returns the snapshot
# closest-but-older than each step — its stale NotImplementedError docstring is wrong):
$PY -m ltn.harvest.ibm_harvester --out data/ibm_live \
    --backfill-start 2026-06-01 --backfill-end 2026-07-01 --backfill-step-hours 24
```

## What one run does

1. Authenticate (`QiskitRuntimeService`; `--instance <CRN>` skips instance discovery)
   — or build the embedded `DryRunService` (2 fake backends x 3 qubits whose
   properties round-trip through the real `qiskit_ibm_runtime.models.BackendProperties`).
2. Enumerate `service.backends(simulator=False, operational=True)`.
3. Per backend: pull `backend.properties(refresh=True)` (or the `datetime=` loop when
   backfilling), **dedup on `(backend_name, last_update_date)`** against both this run
   and files already on disk.
4. Persist each new snapshot twice:
   - raw `props.to_dict()` JSON at `<out>/raw/<backend>/<YYYYMMDDTHHMMSSZ>.json`
     (layout documented in `schemas.py`);
   - UTS events (`ltn.data.uts`, `source="ibm_live"`, `platform="ibm_sc"`,
     `channel="claimed"`) in the hive lake `<out>/uts/`: per-qubit
     `t1_us` / `t2_us` / `frequency_ghz` / `readout_error`, per-coupler
     `gate_error_2q`; `ts` = each parameter's own measurement date,
     `cal_ts` = `last_update_date`; faulty qubit/gate sets in every row's `context`.
5. Exit with a structured-log + JSON summary: `n_backends`, `n_snapshots`, `n_events`,
   plus dedup/missing/error counts. Re-runs are idempotent at both layers (snapshot
   files and UTS row dedup).

HTTP 429 is **not** auto-retried by qiskit-ibm-runtime 0.47.0, so every API call is
wrapped in `with_backoff` (exponential backoff, capped, other errors re-raise; one
failing backend never kills the fleet pull).

## Files

- `ibm_harvester.py` — CLI, backoff wrapper, `DryRunService`, harvest loop.
- `schemas.py` — raw-snapshot JSON layout + pure `from_backend_properties(props_dict,
  backend_name) -> UTS DataFrame` (unit-testable without network).
- `tests/test_harvester_dryrun.py` — dry-run end-to-end, mapping, dedup, backoff.
