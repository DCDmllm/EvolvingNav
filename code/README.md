# EvolvingNav Agent

## Code

| Path | Function |
| --- | --- |
| `src/readyagent/p4d_belief/` | Continuous-time history encoder and persistence–relocation belief |
| `evolvingnav_paper/memory.py` | Causal entity versions, RGB-D backprojection and evidence provenance |
| `evolvingnav_paper/transition_model.py` | Row-normalized chronological transition head |
| `evolvingnav_paper/filter.py` | Current-time belief, arrival forecasts and evidence rounds |
| `evolvingnav_paper/coverage.py`, `calibration.py` | Online depth coverage and validation-fitted detection probability |
| `evolvingnav_paper/agent.py`, `controller.py` | Event-driven actions and frozen VLM tool selection |
| `evolvingnav_paper/world.py`, `backend.py`, `run.py` | Habitat action adapter and benchmark runner |
| `scripts/` | Dataset packing, belief/transition training and calibration |
| `tests/` | Unit tests |

## Environment

Use Python 3.11 with Habitat-Sim 0.3.3 installed.

```bash
cd code
python -m pip install -r requirements.txt
export PYTHONPATH=.:src:scripts
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
export P4D_DATASET=/absolute/path/to/p4d_hssd_30d_v0.6.0
export NAV_TASKS=/absolute/path/to/p4d_navigation_107734254_v1.0
export HSSD_ROOT=/absolute/path/to/hssd-hab
export NAVMESH_ROOT=/absolute/path/to/hssd-hab/navmeshes
```

## Train

If `records/packed/{train,val,test}.npz` are absent:

```bash
python scripts/pack_p4d_hssd_records.py --root "$P4D_DATASET"
```

Train the query-time belief and the chronological transition head:

```bash
python scripts/train_p4d_belief.py \
  --dataset-root "$P4D_DATASET" --output runs/p4d_seed0 \
  --seeds 0 --skip-classical
python scripts/train_transition.py \
  --dataset "$P4D_DATASET" \
  --belief-checkpoint runs/p4d_seed0/checkpoints/p4d/seed_0/best.pt \
  --output runs/transition_seed0
```

Collect held-out RGB-D validation observations and fit detector calibration:

```bash
python scripts/collect_calibration.py \
  --dataset "$P4D_DATASET" --tasks "$NAV_TASKS" \
  --hssd-root "$HSSD_ROOT" --navmesh-root "$NAVMESH_ROOT" \
  --limit 8 --output runs/calibration_val.jsonl
python scripts/fit_calibration.py \
  --validation-jsonl runs/calibration_val.jsonl \
  --output runs/detection_calibration.json
```

## Run

Run the event-driven N3 Agent with Grounding DINO + SAM2:

```bash
python -m evolvingnav_paper.run \
  --task n3 --world static --limit 10 \
  --dataset "$P4D_DATASET" --tasks "$NAV_TASKS" \
  --hssd-root "$HSSD_ROOT" --navmesh-root "$NAVMESH_ROOT" \
  --checkpoint runs/p4d_seed0/checkpoints/p4d/seed_0/best.pt \
  --calibration runs/detection_calibration.json \
  --output runs/n3_static_10
python -m evolvingnav_paper.verify_visual runs/n3_static_10 \
  --tasks "$NAV_TASKS" --hssd-root "$HSSD_ROOT" \
  --navmesh-root "$NAVMESH_ROOT"
```

Add `--controller luna` and set `OPENAI_API_KEY` to use the frozen GPT-5.6-Luna tool controller. Model IDs and revisions for Grounding DINO and SAM2 are in `configs/perception.yaml`.

For an N4 task directory with `public/episodes_n4.jsonl`, each private `target_motion_schedule` event supplies seconds after query (`time_s`), `target_position_xyz`, `current_state_id`, and `valid_goal_viewpoints`:

```bash
python -m evolvingnav_paper.run \
  --task n4 --world routine --limit 2 \
  --dataset "$P4D_DATASET" --tasks /absolute/path/to/n4_tasks \
  --hssd-root "$HSSD_ROOT" --navmesh-root "$NAVMESH_ROOT" \
  --checkpoint runs/p4d_seed0/checkpoints/p4d/seed_0/best.pt \
  --transition-checkpoint runs/transition_seed0/best.pt \
  --output runs/n4_routine_2
```

Every run writes `policy.jsonl`, `scores.jsonl` and `summary.json` to a new output directory.

## Tests

```bash
python -m pytest tests -q
```
