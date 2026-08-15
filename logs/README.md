> **Status: OPERATIONAL / NOT CLAIM-ELIGIBLE (cluster/job snippets).**  
> Do not cite files in `logs/` as the live claim surface. Current authority: `README.md`, `docs/CLAIM_MAP_PREWRITE.md`, `docs/RESULTS_INDEX.md`, `docs/CHERAN_MANUSCRIPT_HANDOFF.md`.  
> These are operational cluster/job snippets (Slurm wrappers, thermal guards, live-status notes), not claim-eligible results.  
> Kept for audit trail. Stale numbers here may be superseded (ToN clean 0.9526 INVALID; principal BoT is 0.9780±0.0033; DICC B3 latency is pre_fix / Option B).  
> Public GitHub visitors: do not promote log text into manuscript tables.

# logs/

Operational run-time notes and cluster helper snippets. **Not** a results archive.

Tracked examples:

- `dicc_d1_live_status.md` — live queue / SUCCESS polling notes
- `job_multi_compiler_{a100,v100s}.sh`, `job_torch_compile_{a100,v100s}.sh` — Slurm job wrappers
- `thermal_guard_neural_baselines.sh` — session thermal-guard helper

Headline numbers belong in `benchmarks/results/` JSON plus `docs/RESULTS_INDEX.md`.
