# legacy — the original scripts, kept for provenance

This folder is what the project looked like before it became a package. It is
kept because the published checkpoint came out of these files, and because the
evolution between them is the actual story: each script is a step in a
deliberate experiment, not four copies of the same thing.

| File | What it is | Outcome |
| :-- | :-- | :-- |
| `first.py` | The baseline. Two-stage training (frozen backbone → full fine-tune), 5 + 5 epochs, Adam, `StepLR`, 64×64 crops from 96×96 resizes. | Works, but the second stage reloads `best.pt` and never resets the epoch counter, so the run is hard to reason about. |
| `second.py` | Same pipeline with the pedagogical comments stripped. | Behaviourally identical to `first.py`. |
| `third.py` | Experiment: `RandomResizedCrop` instead of centre-crop, AMP (`GradScaler`), `ReduceLROnPlateau`, early stopping. | Faster per epoch and smoother loss, but the augmentation change alone costs a few points at 64 px. |
| `fourth.py` | Experiment: 50 + 50 epochs, per-epoch CSV logging and plotting. | **This is the run behind `best.pt`** — 100 epochs, best validation accuracy 0.6956 at epoch 82 (stage 2). |

Also here:

| File | What it is |
| :-- | :-- |
| `training_results_stage1.csv` | Stage-1 log of the 100-epoch run (frozen backbone, lr 1e-2 → 1e-6) |
| `training_results_stage2.csv` | Stage-2 log (unfrozen, lr 1e-3 → 1e-9) |
| `training_results_stageA.csv` | An earlier single-stage 50-epoch run, kept for comparison |
| `myplot*.png`, `training_results_plots*.png` | The plots those scripts drew — `docs/images/` now holds the regenerated versions |
| `flower_result.png` | Sample predictions from `first.py` |
| `OLD_README.md`, `OLD_requirements.txt` | The previous README and dependency list |

Nothing in `src/` imports from this folder. `scripts/import_training_log.py`
reads the two stage CSVs and restates them in the format the current plotting
code expects.
