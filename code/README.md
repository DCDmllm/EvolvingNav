# EvolvingNav N1/N2 code

## Files

| Path | Purpose |
| --- | --- |
| `src/readyagent/p4d_belief/` | Continuous-time history encoder and persistence–relocation belief |
| `scripts/pack_p4d_hssd_records.py` | Pack query records into model tensors |
| `scripts/train_p4d_belief.py` | Train and save belief checkpoints |
| `evolvingnav_paper/policy.py` | Public-query packing and N1/N2 candidate selection |
| `evolvingnav_paper/perception.py` | Frozen Grounding DINO + SAM2 inspection |
| `evolvingnav_paper/backend.py` | Habitat rendering and geodesic paths |
| `evolvingnav_paper/evaluate.py` | STOP, success and SPL evaluation |
| `evolvingnav_paper/run.py` | N1/N2 episode runner |
| `evolvingnav_paper/verify_visual.py` | Independent visual and viewpoint check |
| `tests/` | Unit tests |

## Run

Use Python 3.11 with Habitat-Sim 0.3.3 installed. Install the remaining Python dependencies in that environment:

```bash
cd code
python -m pip install -r requirements.txt
export PYTHONPATH=.:src:scripts
export MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet
```

Set absolute paths to the prepared belief dataset, navigation task dataset, HSSD scene assets and NavMesh cache:

```bash
export P4D_DATASET=/absolute/path/to/p4d_hssd_30d_v0.6.0
export NAV_TASKS=/absolute/path/to/p4d_navigation_107734254_v1.0
export HSSD_ROOT=/absolute/path/to/hssd-hab
export NAVMESH_ROOT=/absolute/path/to/navmeshes
```

`P4D_DATASET` contains `records/packed/train.npz`, `val.npz` and `feature_schema.json`. `NAV_TASKS` contains `public/episodes_n2.jsonl`, `public/query_inputs.jsonl`, `private/evaluation_gt.jsonl` and `catalogs/`.

```bash
python -m pytest tests -q
python scripts/train_p4d_belief.py \
  --dataset-root "$P4D_DATASET" --output runs/p4d_seed0 \
  --seeds 0 --skip-classical
python -m evolvingnav_paper.run \
  --task n2 --world static --limit 2 \
  --dataset "$P4D_DATASET" --tasks "$NAV_TASKS" \
  --hssd-root "$HSSD_ROOT" --navmesh-root "$NAVMESH_ROOT" \
  --checkpoint runs/p4d_seed0/checkpoints/p4d/seed_0/best.pt \
  --output runs/n2_static_2
python -m evolvingnav_paper.verify_visual runs/n2_static_2 \
  --tasks "$NAV_TASKS" --hssd-root "$HSSD_ROOT" \
  --navmesh-root "$NAVMESH_ROOT"
```

The perception model IDs and pinned revisions are in `configs/perception.yaml`. To use local model directories, pass `--grounding-dino-model /path/to/model` and `--sam2-model /path/to/model` to `evolvingnav_paper.run`.

Each evaluation run uses a new output directory and writes `policy.jsonl`, `scores.jsonl` and `summary.json`. `verify_visual` writes `visual_check.json` into that directory.
