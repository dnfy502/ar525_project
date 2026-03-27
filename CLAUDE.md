# Claude Code Rules — AR525 MC-PILOT Project

## Change Tracking (MANDATORY)

**Every code change made to this repo must be logged in `change_history.md`.**

- Before making any change, check `change_history.md` to understand prior context and avoid repeating failed experiments.
- After making any change (no matter how small), add an entry under the appropriate section:
  - Bug fix → under "Baseline Bugs Fixed"
  - Performance improvement → under "Performance Improvements"
  - Failed experiment → under "Experiments That Did NOT Work" (include what was tried, what happened, root cause, lesson)
  - Update "Current Best Configuration" table and "Current Best Scores" if results changed.
- Entry format: heading, files touched, problem description, fix/change, result.

## Result Interpretation (MANDATORY)

**When the user shares experimental results (hit rates, cost values, errors), always:**

1. Interpret them immediately — compare against current best, identify patterns (near vs far targets, cost trajectory, etc.)
2. Update `change_history.md` with the result and interpretation under the relevant experiment entry.
3. Update `current_config.md` results table if the overall hit rate changed.
4. Flag if the result is better/worse than baseline (5/10 hits, seed=1) and why.

Do not wait for the user to ask — record results as soon as they are shared.

## Results Files

- Results are saved to `mc-pilot/results_mc_pilot/<seed>/` (e.g. seed=1 → `results_mc_pilot/1/`)
- `config_log.pkl` — hyperparameters used for that run
- `log.pkl` — full log: `noiseless_states_history` (all throw trajectories), `cost_trial_list` (cost per opt step per trial), `state_samples_history`, etc.
- To read results: load both pkl files with Python using the project venv. Extract landing errors from `noiseless_states_history` final states (ball xy = dims 0:1, target xy = dims 6:7).

## General

- The working venv is at `/home/dnfy/Desktop/AR525 RL/project/.venv`. Always use this for running Python.
- Never rerun a previously failed experiment without first explaining why the outcome will differ.
- Before suggesting a change, check `change_history.md` to confirm it hasn't already been tried and documented as failed.
