# Issue #265 implementation guide

## What is implemented

`adaptive-rl benchmark adaptation` runs the protocol's train-once, shared
pre-shift, shared shock, fork, and paired post-shift lifecycle for drone TEST-B
with PPO or SAC. The base environment receives only the nominal configuration
from `configs/drone_distribution_shift.yaml`; the TEST-B overrides are merged
after training and the pre-shift segment. The run checks the effective shift
values before proceeding.

The online data boundary is represented by completed post-shift episode
records. Block Bk receives the ordered prefix 1..k only, and the builder rejects
missing, duplicated, out-of-order, wrong-seed, or future episode data. Blocks
B5 through B14 execute between episodes; B15 is rejected. PPO uses stored
behavior values/log-probabilities with its native clipped update. SAC uses a
new replay buffer containing only the allowed post-shift prefix. Both use the
block's derived update seed. Update failures roll model state back and fail the
replicate.

The fork deep-copies both arms, verifies equal starting policy states, and gives
the Fixed arm a prediction-only interface. The runner verifies its final
fingerprint against the frozen fingerprint. Episode records include outcomes,
derived seeds, policy fingerprints, update association, and the complete
transition data used by adaptation.

Recovery is computed only by
`adaptive_rl.protocol.recovery.compute_recovery()`. Paired analysis uses the
protocol's paired t-test, interval, dz, sign and Wilcoxon sensitivities,
bootstrap interval, failure-imputation bounds, and Holm correction. Since this
checkout implements the selected drone cell only, the other five primary cells are
reported as inconclusive in each single-cell artifact; the six-cell family
claim is therefore inconclusive until the remaining cells have data.

## Running it

Run a reduced, explicitly labeled machinery check:

```bash
adaptive-rl benchmark adaptation \
  --config configs/drone_distribution_shift.yaml \
  --smoke \
  --output-dir artifacts/issue265_smoke_ppo
```

Run the smoke path for SAC:

```bash
adaptive-rl benchmark adaptation \
  --config configs/drone_distribution_shift.yaml \
  --algorithm sac \
  --smoke \
  --output-dir artifacts/issue265_smoke_sac
```

Run the complete ten-seed PPO experiment (the same command accepts `--algorithm
sac` for SAC):

```bash
adaptive-rl benchmark adaptation \
  --config configs/drone_distribution_shift.yaml \
  --output-dir artifacts/issue265_ppo
```

`--training-seeds` may select a comma-separated subset of the frozen training
seeds. Unselected replicates remain missing and count as failures for
pair-complete analysis. `--deterministic` and `--stochastic` override the
configured evaluation action selection. The command writes
`adaptation.json` and `adaptation.csv`; it refuses to overwrite either file or
an existing per-seed training directory.

The JSON contains the full trajectories, replicate failures and reasons,
protocol and schedule fingerprints, Treatment Card SHA-256, canonical and raw
configuration hashes, repository commit/dirty state, runtime versions,
recovery outputs, and paired analysis. CSV has one row per recorded episode.
Smoke output is labeled `run_type: smoke`, uses a reduced training budget and
episode length, and is never treated as a scientific result.

## Reproducibility and status

Adaptation choices are frozen in
[`TREATMENT_CARD.md`](TREATMENT_CARD.md). The JSON records its content hash,
the full derived schedule fingerprint, each replicate's exact phase seeds,
each update seed, and model fingerprints before and after every block. Failed
training, evaluation, or update work is represented in `failure_summary` and
the replicate record; a failed update stops that replicate before the next
episode.

Both PPO and SAC smoke paths have been executed and checked for block order,
episode counts, seed alignment, Fixed immutability, and recovery output. Those
checks validate the harness only. No full ten-replicate benchmark has been run
and no empirical superiority or statistical significance is claimed.
