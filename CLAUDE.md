# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research pipeline for surgical action triplet recognition (`<instrument, verb, target>`) on the
CholecT50 dataset, built on top of CAMMA's **Rendezvous** model (Nwoye et al., 2022). It is not the
original CAMMA repo as-is: the base model has been extended into a custom multi-stage
"specialist → fusion → temporal → spatio-temporal fusion head" pipeline that is not described in the
original paper. There is no test suite, linter, or CI — this is an experimental ML codebase driven by
standalone scripts and inspected via logs/plots/mAP numbers.

Code comments, print statements, and many identifiers are in Spanish; keep that convention when editing
existing scripts.

## Environment

- A Python venv is committed at `venv/`. Activate it (`venv\Scripts\Activate.ps1` on PowerShell) before
  running anything, or use its interpreter directly.
- `requirements.txt` lists the core deps: `torch`/`torchvision`, `numpy`, `opencv-python`, `Pillow`,
  `matplotlib`/`seaborn`, and `ivtmetrics==0.1.5` (the official CholecT50 mAP metric).
- Everything assumes a CUDA GPU is available (`.cuda()` is called unconditionally in most training
  scripts); `evaluador_map.py` is the exception and falls back to CPU.

## Repository layout

- `src/` — canonical, current model code: `network.py` (Rendezvous), `tcn.py` (CirugiaTCN temporal
  model), `spatial_temporal_rdv.py` (spatio-temporal fusion module), `dataloader.py` (CholecT50 dataset/
  split logic, copied from the original CAMMA repo).
- `scripts/` — every pipeline stage and inference/visualization entry point (see below). `scripts/legacy/`
  holds old diagnostic one-offs; treat as archived, not part of the active pipeline.
- `rendezvous/` — vendored copy of the original CAMMA Rendezvous repo (LICENSE, README, `original/`).
  `rendezvous/pytorch/__checkpoint__/` holds training-run outputs from when the specialist scripts were
  executed with that directory as the working directory.
- `data/CholecT50/` — the dataset in official CAMMA format (`instrument/`, `verb/`, `target/`,
  `triplet/`, `labels/` (JSON per video), `videos/` (extracted frames)). `data/features_1d_custom/` and
  `data/cams_offline/` hold derived artifacts produced by the feature/CAM extraction scripts.
- `checkpoints/` — the weights actually consumed by the downstream (Phase 3+) scripts:
  `run_organos/`, `run_verbos/`, `run_instrumentos/`, `run_fusion/` (mirrors of the corresponding
  `rendezvous/pytorch/__checkpoint__/run_*` outputs — best checkpoints are copied here manually after
  training) plus `TCN_FINAL.pth` and `SpatioTemporal_FINAL.pth`.
- `plots/`, `Legacy/` — generated figures/confusion matrices and archived earlier experiments,
  respectively. Not inputs to the pipeline.

## The pipeline, in order

This is the load-bearing thing to understand: the final triplet prediction is **not** just the
Rendezvous model's own `dec_ivt` output. It's a separate spatio-temporal head trained on top of frozen,
pre-extracted Rendezvous features. Each phase's output is a checkpoint/artifact consumed by the next.

1. **Per-component specialists** (`scripts/run_instrumentos.py`, `run_organos.py`, `run_verbos.py`) —
   fine-tune a full `Rendezvous('resnet18')` per component (instrument/verb/target), unfreezing the
   ResNet18 backbone. Saves to `./__checkpoint__/run_{name}_{version}/` relative to cwd.
2. **Fusion (Phase 4 in the code's own terms)** (`scripts/run_fusion.py`) — builds one `Rendezvous`
   model, injects the organs specialist's weights (base + bottleneck) and the verbs specialist's CAGAM
   weights, then unfreezes and fine-tunes everything end-to-end with a very small LR. Saves to
   `./__checkpoint__/run_fusion_{version}/Rendezvous_FUSION_FINAL.pth`. The best checkpoint from this
   step is copied to `checkpoints/run_fusion/Rendezvous_FUSION_FINAL.pth`, which every later phase reads.
3. **Offline feature/CAM extraction** (`scripts/extract_features.py`, `scripts/extract_cams.py`) — run
   the frozen fusion checkpoint once over every video frame. `extract_features.py` hooks
   `encoder.basemodel.basemodel.layer4[1].bn2`, global-average-pools it to a 512-d vector per frame, and
   writes `data/features_1d_custom/VIDxx.npy` (`[T, 512]`). `extract_cams.py` captures the raw
   per-component CAMs (`[T, 6/10/15, 7, 7]`) and writes `data/cams_offline/cams_{inst,verb,targ}_VIDxx.npy`.
   This step exists so later training never touches raw images again.
4. **Temporal model — "Fase B"** (`scripts/train_temporal.py`) — trains `CirugiaTCN`
   (`src/tcn.py`, a 4-block dilated 1D TCN with *bidirectional/non-causal* padding, since CholecT50 videos
   are offline recordings) on the extracted 512-d features against per-frame labels parsed from
   `data/CholecT50/labels/*.json`. Produces per-class logits (instrument/verb/target/triplet), used only
   as a temporal *confidence signal*, not as the final classifier. Output: `checkpoints/TCN_FINAL.pth`.
5. **Spatio-temporal fusion head — "Fase C"** (`scripts/train_fusion_head.py`) — trains
   `FusionHeadTrainer`, which for each component runs a `SpatialRefinementModule`: modulate the 7×7 CAM
   by `sigmoid(TCN logit)` as a per-class confidence mask, concat with the original CAM, refine with a
   small conv, and add back a learned residual (deliberately **no final ReLU** — CAM values are mostly
   negative, and an earlier version's ReLU there was killing gradient and stalling loss at `ln(2)`). The
   three refined component maps (31 channels total) feed a `triplet_head` conv producing the 100 final
   triplet logits. Output: `checkpoints/SpatioTemporal_FINAL.pth`.
6. **Official evaluation** (`evaluador_map.py`, repo root) — loads the fusion head + TCN, runs inference
   over the `cholect50-crossval` fold-1 test videos, and computes mAP with `ivtmetrics` (`ignore_null=False`),
   printing results against the paper's reference numbers. `scripts/evaluador_rdv_puro.py` is the
   comparison baseline: it evaluates the raw fused Rendezvous model without the TCN/fusion-head stages.
7. **Inference/visualization** (not part of training) — `scripts/inferencia_visual.py`,
   `inferencia_visualizacion.py`, `generar_video.py`, `{Instrument,Target,Verb,triplets}plot_sptlRDV.py`,
   and the `generador_matriz*`/`generacion_matrices_*` scripts render heatmaps, demo videos, and confusion
   matrices into `plots/`.

## Known gotchas when touching scripts

- **Import style is inconsistent between phases.** `scripts/run.py`, `run_instrumentos.py`,
  `run_organos.py`, `run_verbos.py`, `run_fusion.py` do bare `import network` / `import dataloader` with
  no `sys.path` manipulation — they expect those modules importable directly (historically run with cwd
  inside `rendezvous/pytorch/`, or with `src/` added to `PYTHONPATH`). The newer Phase 3+ scripts
  (`extract_features.py`, `extract_cams.py`, `train_temporal.py`, `train_fusion_head.py`,
  `evaluador_map.py`) instead explicitly `sys.path.append(root_dir)` and import `src.network`/`src.tcn`.
  Don't assume one import style works for both groups.
- **`root_dir` is hardcoded to `r"C:\ProyectoRDV"` in some scripts** (`evaluador_map.py`,
  `extract_cams.py`, `train_fusion_head.py`) but computed dynamically via `__file__` in others
  (`extract_features.py`, `train_temporal.py`). If the repo is ever relocated, the hardcoded ones will
  silently point at the wrong place.
- **`FusionHeadTrainer`/`SpatialRefinementModule` are redefined inline** in both
  `train_fusion_head.py` and `evaluador_map.py` rather than imported from `src/spatial_temporal_rdv.py`.
  A fix to the fusion-head architecture needs to be applied in all three places (and kept
  bit-for-bit identical between train and eval, or checkpoints won't load / results won't match).
- **Two parallel checkpoint trees exist**: `rendezvous/pytorch/__checkpoint__/run_*` (raw training-script
  output) and `checkpoints/run_*` (repo-root copies actually consumed downstream). Promoting a new
  specialist/fusion checkpoint requires manually copying it across.
- The `evaluador_map.py` test split (`SPLITS["cholect50-crossval"]`) is a hardcoded duplicate of the
  split logic in `src/dataloader.py`'s `CholecT50.split_selector` — if the official split ever changes
  there, update both.
- `ResNet18` backbone is **frozen by default** in `src/network.py`'s `BaseModel` (`requires_grad=False`
  set at construction); specialist scripts that need end-to-end fine-tuning explicitly re-enable
  `requires_grad` afterward.

## Running things

There's no build/lint/test command — validate changes by running the relevant script and checking its
printed loss/mAP, or by running `evaluador_map.py`/`scripts/evaluador_rdv_puro.py` end to end and
comparing against the reference mAP values it prints (AP_I 92.0 / AP_V 60.7 / AP_T 38.3 / mAP_IVT 29.9,
per Nwoye et al. 2022).

Typical invocation shape (flags vary per script — check each script's `argparse` block):

```
python scripts/run_instrumentos.py -t --data_dir="data/CholecT50" --dataset_variant=cholect50-crossval -k 1
python scripts/run_fusion.py -t --data_dir="data/CholecT50" --dataset_variant=cholect50-crossval -k 1
python scripts/extract_features.py
python scripts/extract_cams.py
python scripts/train_temporal.py
python scripts/train_fusion_head.py
python evaluador_map.py
```

All training scripts support resuming from their own checkpoint automatically (they check
`os.path.exists(ckpt_path)` and reload epoch/optimizer/scheduler state), so re-running the same command
after an interruption continues rather than restarting.
