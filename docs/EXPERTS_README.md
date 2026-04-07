# Experts README

## Purpose

This document explains the expert side of the `TPUGPU` POC end to end:

- what model is being trained
- what objective it uses
- how the strict label split works
- where the TPU and GPU experts run
- what artifacts are produced

This is the authoritative description of the current expert workflow.

## Current Expert Model

The current expert is:

- dataset: `MNIST`
- architecture: small class-conditional `UNet`
- framework: `JAX`
- conditioning: class label `y in {0..9}`
- objective: `flow matching`

The expert predicts a velocity field, not DDPM epsilon.

Core implementation:

- [model.py](/Users/ali/projects/TPUGPU/src/tpugpu/experts/model.py)
- [train.py](/Users/ali/projects/TPUGPU/src/tpugpu/experts/train.py)
- [train_expert_mnist.py](/Users/ali/projects/TPUGPU/scripts/train_expert_mnist.py)

## Flow-Matching Objective

Each training example is built from:

- clean image `x_0`
- Gaussian noise `x_1`
- timestep `t ~ Uniform(0, 1)`

The noisy state is:

`x_t = (1 - t) * x_1 + t * x_0`

The target velocity is:

`v* = x_0 - x_1`

The model is trained to predict that velocity from:

`(x_t, t, y)`

So the expert contract is:

- input: `x_t, t, y`
- output: `velocity`

That exact same contract is used later for distributed inference.

## Class Conditioning

The expert is class-conditional, not text-conditional.

The model receives:

- noisy image `x_t`
- timestep `t`
- class label `y`

The class label is passed through a learned embedding table, not a text encoder.

Important implication:

- the model can accept any valid label `0..9`
- even if a specific expert never trained on that label's images

So the model does not throw a runtime error on held-out labels. It just performs badly on them.

## Strict Split

The current strict split design is:

- expert A: labels `0,1,2,3,4`
- expert B: labels `5,6,7,8,9`

The train/test split itself is still standard MNIST:

- training: 60,000 examples
- test: 10,000 examples

Then the repo filters by class IDs.

Dataset logic:

- [mnist.py](/Users/ali/projects/TPUGPU/src/tpugpu/data/mnist.py)

## TPU Expert

Current TPU expert:

- VM: `tpugpu-v6e-1-e5a`
- zone: `us-east5-a`
- accelerator: `v6e-1`
- backend: `tpu`
- expert name: `expert_strict_0_4_fm`
- labels: `0,1,2,3,4`

Checkpoint location on TPU VM:

- `/home/ali/TPUGPU/outputs/checkpoints/expert_strict_0_4_fm`

Downloaded local artifacts:

- `/Users/ali/Desktop/TPUGPU_tpu_strict_0_4`
- `/Users/ali/Desktop/TPUGPU_tpu_strict_0_4_heldout`

## GPU Experts

There are two GPU-related expert runs in this project history.

### L4 comparison run

- VM: `tpugpu-gpu-l4-e4a`
- zone: `us-east4-a`
- GPU: `NVIDIA L4`
- backend: `gpu`
- expert name: `expert_strict_0_4_gpu_fm_b16`
- labels: `0,1,2,3,4`

This run was useful for TPU vs GPU parity on the same task.

Downloaded local artifacts:

- `/Users/ali/Desktop/TPUGPU_gpu_strict_0_4`
- `/Users/ali/Desktop/TPUGPU_gpu_strict_0_4_heldout`

### Far-region A100 expert

- VM: `tpugpu-gpu-a100-sg1a`
- zone: `asia-southeast1-a`
- GPU: `NVIDIA A100 40GB`
- backend: `gpu`
- target expert name: `expert_strict_5_9_gpu_fm_a100`
- labels: `5,6,7,8,9`

This is the complementary expert intended for the real routed system.

At the time of writing:

- training is running or being retried under smaller batch settings
- the blocker has been evaluation-time OOM, not the main training loop

## Training Command Pattern

General expert training command:

```bash
python scripts/train_expert_mnist.py \
  --expert-name expert_name_here \
  --class-ids 0,1,2,3,4 \
  --num-epochs 50 \
  --batch-size 64 \
  --sample-every-epochs 5 \
  --eval-num-generated 512 \
  --eval-batch-size 64
```

Important note:

- training batch size and eval batch size are separate
- eval can OOM even when training is stable

## Artifacts

Each expert run writes:

- checkpoint directory
- `train.log`
- per-epoch generated grids
- per-epoch label histograms
- per-epoch t-SNE plots
- training curves
- `history.json`

Typical structure:

- `outputs/checkpoints/<expert_name>/`
- `outputs/experiments/<expert_name>/`

## Held-Out Label Test

The strict split was used to prove this specific point:

- the model accepts held-out labels inside the valid label space
- but generation quality degrades badly

That test was run with:

- TPU `0..4` expert, forced to generate `5..9`
- GPU `0..4` expert, forced to generate `5..9`

Inference script:

- [sample_expert_mnist.py](/Users/ali/projects/TPUGPU/scripts/sample_expert_mnist.py)

## End-to-End Expert Flow

The correct workflow is:

1. Edit locally in `/Users/ali/projects/TPUGPU`
2. Commit locally
3. Push to GitHub
4. Pull on the target VM
5. Run training on that VM
6. Download artifacts locally

That workflow has been followed consistently for the project after the initial bring-up.
