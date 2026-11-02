"""Token-gated stub. Live IBM calibration harvesting is not used in this work [DEV-1].

The complete, dry-run-tested harvester lives in ltn/harvest/ibm_harvester.py and would
populate source="ibm_live" the day an IBM Quantum token exists. This stub exists so the
loader registry is total over ltn.data.uts.SOURCES.
"""


def load_ibm_live(raw_dir=None):
    raise RuntimeError(
        "source='ibm_live' is token-gated and unused in this work [DEV-1]; "
        "see ltn/harvest/README.md"
    )
