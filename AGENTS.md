# YOLOX Agent Instructions

## Project
YOLOX v0.3.0 — anchor-free object detection in PyTorch. Single repo, no monorepo layout.

## Install
```
pip3 install -v -e .
```
The `setup.py` conditionally compiles a C++ extension (`FastCOCOEvalOp`) in `yolox/layers/` on non-Windows when torch is importable. If torch is missing the build silently skips the extension.

## Key entry points

| Purpose | Command |
| --- | --- |
| Train | `python -m yolox.tools.train -n yolox-s -d 8 -b 64 --fp16 -o` |
| Eval | `python -m yolox.tools.eval -n yolox-s -c yolox_s.pth -b 64 -d 8 [--fp16] [--fuse]` |
| Demo (image) | `python tools/demo.py image -n yolox-s -c <ckpt> --path <img> --conf 0.25 --nms 0.45 --tsize 640 --save_result --device [cpu/gpu]` |
| Demo (video) | same as above with `video` instead of `image` |
| Export ONNX | `python tools/export_onnx.py` |
| Export TorchScript | `python tools/export_torchscript.py` |
| Export TensorRT | `python tools/trt.py` |
| Visualize assignments | `python tools/visualize_assign.py` |

`-n` (name) or `-f` (exp file) are required. Names map to `yolox.exp.default.*` modules (`yolox-s` → `yolox_s`). Using `-f` accepts any path with a top-level `Exp` class.

## Experiment configs
Experiments are single files in `exps/default/` and `exps/example/`. Each defines a class `Exp` inheriting from `yolox.exp.Exp` (which itself extends `BaseExp`). Subclass only overrides what changes — depth, width, input_size, data_dir, etc. The base class holds defaults for multiscale, augmentation probs, LR schedule, evaluator, etc.

To add a new model variant: create a file in `exps/default/`, define `class Exp(yolox.exp.Exp)`, override fields. No registration step needed.

## Dataset setup
```
ln -s /path/to/COCO ./datasets/COCO
```
Annotation files expected: `instances_train2017.json`, `instances_val2017.json`, `instances_test2017.json`. Override the dataset root with env var `YOLOX_DATADIR`.

## Output
Default output directory: `./YOLOX_outputs`. Controlled by `Exp.output_dir`.

## Distributed training
Uses `torch.distributed` via `yolox.core.launch`. Flags: `-d` (device count), `--dist-backend` (default `nccl`), `--dist-url`, `--num_machines`, `--machine_rank`. The launcher auto-finds a free port if `--dist-url` is not given.

## Training quirks
- `--fp16` enables `torch.cuda.amp.GradScaler` — not optional mixed-precision, it is required for the documented speed numbers.
- `--cache` loads images into RAM via `DataPrefetcher` before training (needs large system RAM).
- `-b` is total batch size across all GPUs; recommended `num_gpus * 8`.
- Multiscale training range: `Exp.multiscale_range` (default 5 → actual range `[640 - 5*32, 640 + 5*32]`). Set to 0 to disable.
- W&B logging: `--logger wandb wandb-project <name>`.

## Evaluation quirks
- `--fuse` fuses Conv+BatchNorm (must be done before export/eval for correct speed numbers).
- `--conf 0.001` is the documented default for speed benchmarks; use higher values for meaningful AP.
- `-b` is total batch across all GPUs. `-d` defaults to all available GPUs.

## Code style
Configured in `setup.cfg`:
- `flake8`: max-line-length 100, max-complexity 18, excludes `__init__.py`
- `isort`: line_length 100, sections = FUTURE, STDLIB, THIRDPARTY, data_processing, datasets, deeplearning, myself, FIRSTPARTY, LOCALFOLDER. `yolox` is its own section.
- Pre-commit hooks (`.pre-commit-config.yaml`): flake8, isort, autoflake (removes unused imports/vars), gitlint, plus standard pre-commit-hooks.

## Format check (CI-equivalent)
```
flake8 yolox exps tools
isort --check-only -rc yolox exps
```
Run this before committing. CI runs the same check (no unit tests executed).

## Testing
Tests are minimal: only `tests/utils/test_model_utils.py` (unittest). No test runner config, no pytest. Run with:
```
python -m unittest tests/utils/test_model_utils.py
```
Do not assume a test suite exists for other modules.

## Package boundaries
- `yolox/core/` — `Trainer`, `launch` (DDP wrapper)
- `yolox/data/` — datasets (COCO, VOC), dataloader, augmentation (`data_augment.py`), prefetcher
- `yolox/models/` — `yolox.py` (top-level model), `yolo_fpn.py`, `yolo_pafpn.py`, `yolo_head.py`, `darknet.py`, `losses.py`, `network_blocks.py`, `build.py`
- `yolox/exp/` — `BaseExp`, `Exp` (yolox_base), `build.py` (get_exp by name/file)
- `yolox/evaluators/` — COCO evaluator, VOC evaluator
- `yolox/utils/` — utilities (dist, checkpoint, lr_scheduler, model_utils, demo_utils, visualize, etc.)
- `yolox/layers/` — custom ops: `FastCOCOEvalOp` (C++ extension), JIT ops (`jit_ops.py`)
- `tools/` (root) — CLI entry points (train, eval, demo, export_*, trt, visualize_assign)
- `exps/` — experiment config files
