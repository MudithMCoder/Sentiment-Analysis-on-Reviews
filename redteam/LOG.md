# Red-team log

## 2026-11-16 — Phase 1: threat model + attack set

- Model: `v0-baseline` (`artifacts/v0`), test accuracy 0.8595.
- Added `THREAT_MODEL.md`: system description, ReelRate scenario, assets, adversaries A1/A2,
  threats M1–M6 as hypotheses, out-of-scope items, metrics and protocol.
- Added `redteam/make_attack_set.py` and built `redteam/attack_set.json`:
  - Command: `python -m redteam.make_attack_set` (seed 42, 250 per class).
  - v0 correct on test split: 3,257 negative, 3,136 positive (of 7,438).
  - Attack set: 500 reviews, 250 negative + 250 positive.
  - Longer than 300 pipeline tokens: 103 / 500 (44 negative, 59 positive). These are the M1 inputs.
  - Median pipeline-token length: 172.
  - A rebuild produces a byte-identical file.
- No attacks run. No results claimed.
