# Light-DR PPO 16-update result

## Training

- parent: B64 handload checkpoint;
- full actor/critic/Adam warm start;
- 128 RSI worlds, 32 steps/update;
- 16/16 accepted updates, 65,536 environment steps;
- wall time: 492.5 s;
- no safety stop;
- friction sampled in `[0.90, 1.10]`;
- PD gains sampled in `[0.95, 1.05]`;
- initial root position `+-1 cm`;
- initial joint angle `+-2 deg`;
- no extreme push, slope, jump, or terrain change.

Mass randomization was not silently approximated. The shared MuJoCo Warp model
does not expose a per-world mass field, so mass DR remains deferred.

## Native screen

Each case ran 17 s from the same real local-support state.

| Case | Crossing | Seat final | Feet L/R | Unapproved T1 Fn | Hard guard |
|---|---:|---:|---:|---:|---:|
| Parent nominal | 4.94 s | 51.1 N | 8.5 / 35.3 N | 89.0 N | none |
| Candidate nominal | 4.90 s | 31.0 N | 8.6 / 36.7 N | 90.4 N | none |
| Parent DR seed 1 | 5.04 s | 35.1 N | 21.0 / 0 N | 149.8 N | none |
| Candidate DR seed 1 | 5.10 s | 29.9 N | 22.8 / 0 N | 159.3 N | none |
| Parent DR seed 2 | 4.76 s | 0 N | 41.4 / 0 N | 207.3 N | none |
| Candidate DR seed 2 | 4.74 s | 0 N | 51.6 / 0 N | 133.9 N | none |
| Parent DR seed 3 | 4.92 s | 28.0 N | 17.7 / 0 N | 165.5 N | none |
| Candidate DR seed 3 | 4.90 s | 0 N | 32.7 / 0 N | 196.8 N | none |

## Decision

**Rejected, no promotion.**

The candidate preserves crossing and avoids hard guards, but the nominal seat
load falls from about `51.1 N` to `31.0 N`; two perturbed seeds lose seat support
entirely, and illegal/unapproved T1 contact remains high. Reward movement and
accepted updates are not treated as success.

The protected handload actor remains the Demo model. Formal legal hold remains
`0 s`.

Rejected-candidate native video:
`assets/light_dr_ppo16_rejected_nominal.mp4`.

## Artifacts

- `checkpoints/light_dr_ppo16_actor.ts`
- `checkpoints/light_dr_ppo16_checkpoint.pt`
- `configs/light_dr_ppo16.json`
- `docs/light_dr_ppo16/training_config.json`
- `docs/light_dr_ppo16/DONE.json`
- `docs/light_dr_ppo16/metrics.jsonl`
- `docs/light_dr_ppo16/native_screen.json`
