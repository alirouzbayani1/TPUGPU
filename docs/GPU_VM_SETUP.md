# GPU VM Setup

## Purpose

This document records the exact GPU VM path for `TPUGPU` so the GPU side follows the same discipline as the TPU side:

1. edit locally
2. push to GitHub
3. pull on the VM
4. run on the VM
5. repeat

The goal is not to create a generic CUDA workstation. The goal is to provision a single-GPU machine that can run the **same model code** as the TPU side for this repo.

## Working GPU Choice

For this project stage, the right GPU VM is:

- zone: `us-east4-a`
- machine type: `g2-standard-4`
- accelerator: `1 x NVIDIA L4`
- image project: `deeplearning-platform-release`
- image family: `common-cu128-ubuntu-2204-nvidia-570`

Why this choice:

- one GPU keeps the setup symmetrical with the current `v6e-1` TPU proof
- `L4` is enough for the small MNIST diffusion model
- `g2-standard-4` is the cheapest clean accelerator-optimized single-GPU option available in the checked zones
- the Deep Learning VM image already includes the NVIDIA driver and CUDA stack, which reduces setup drift

This is not the final production GPU choice. It is the correct choice for the current proof.

## Requirements

Before creating the GPU VM, verify:

```bash
gcloud config get-value account
gcloud config get-value project
```

Expected for the current documented setup:

- account: `ali@carezai.com`
- project: `ir-dicompoc`

## Create The GPU VM

Use:

```bash
gcloud compute instances create tpugpu-gpu-l4-e4a \
  --zone=us-east4-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --image-project=deeplearning-platform-release \
  --image-family=common-cu128-ubuntu-2204-nvidia-570 \
  --boot-disk-size=200GB
```

Important notes:

- `g2-standard-4` implies one `L4`
- `TERMINATE` is required for GPU VMs
- the DLVM image is preferred over a plain Ubuntu image because it reduces driver/CUDA setup risk

## SSH Into The GPU VM

Use:

```bash
gcloud compute ssh tpugpu-gpu-l4-e4a --zone=us-east4-a
```

## Initial Inspection

After SSH:

```bash
python3 --version
lsb_release -a
free -h
nvidia-smi
```

`nvidia-smi` is the GPU-side equivalent of the TPU verification commands.

## GitHub Auth And Repo Clone

Install `gh` if needed:

```bash
sudo apt-get update
sudo apt-get install -y gh
```

Then:

```bash
gh auth login
git clone https://github.com/bageldotcom/TPUGPU.git
cd TPUGPU
```

## Bootstrap The Repo Environment

Run:

```bash
bash scripts/setup_gpu_vm.sh
```

This creates:

- virtualenv: `~/tpugpu-gpu-venv`

Activate later with:

```bash
source ~/tpugpu-gpu-venv/bin/activate
```

## What The Script Installs

The GPU setup script installs:

- `jax[cuda12]`
- `flax`
- `matplotlib`
- `optax`
- `scikit-learn`
- `tensorflow`
- `tensorflow-datasets`
- `orbax-checkpoint`
- `chex`
- `einops`
- `pytest`
- `rich`

and then installs the repo in editable mode.

## Verification

After setup:

```bash
source ~/tpugpu-gpu-venv/bin/activate
python -c "import jax; print(jax.devices()); print(jax.default_backend())"
nvidia-smi
```

Expected:

- JAX sees a GPU device
- backend is `gpu`
- `nvidia-smi` reports one `L4`

## Project Rule

Do not patch source code directly on the GPU VM unless absolutely necessary.

The correct loop is:

1. edit locally in `/Users/ali/projects/TPUGPU`
2. commit locally
3. push to Bagel GitHub
4. pull on the GPU VM
5. run on the GPU VM
6. collect artifacts

This keeps TPU and GPU execution consistent and makes comparisons defensible.
