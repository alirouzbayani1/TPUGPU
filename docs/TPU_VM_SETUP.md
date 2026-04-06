# TPU VM Setup

## Purpose

This document records the exact process used to get a working `JAX/XLA` development environment on a real Google Cloud `TPU VM` for this repo.

The goal is to make TPU bring-up repeatable. The next time we need a fresh TPU VM, we should be able to follow this document and the repo bootstrap script instead of rediscovering:

- which TPU type to request
- which runtime version to use
- which Google account/project was used
- which packages were missing on the base image
- which verification commands actually prove the machine is usable

This document is intentionally practical. It describes the setup that actually worked for `TPUGPU`, not an abstract TPU tutorial.

## Working Configuration

This is the configuration that successfully booted and ran the first TPU-side MNIST expert training job for this repo.

- Google account: `ali@carezai.com`
- GCP project: `ir-dicompoc`
- TPU VM name: `tpugpu-v6e-1-e5a`
- Zone: `us-east5-a`
- Accelerator type: `v6e-1`
- TPU generation: `Trillium / v6e`
- Runtime version: `v2-alpha-tpuv6e`

This is a small `single-chip` TPU VM. It is not intended to be the final large training machine. It is the smallest modern TPU configuration that actually provisioned successfully for this project and was enough to:

- install the TPU JAX stack
- run JAX on TPU
- train the first MNIST expert
- write a checkpoint successfully

## Why This TPU Choice

For this project stage, we needed:

- a real TPU runtime
- the fastest path to a usable machine
- a small footprint for JAX/XLA learning and MNIST-scale work

We did **not** need:

- a large training slice
- `v5p`
- a multi-host topology

Several larger or alternate TPU attempts failed because of capacity or permissions. The first configuration that actually worked was:

- `v6e-1`
- in `us-east5-a`
- with `v2-alpha-tpuv6e`

That is why this document standardizes on that configuration for now.

## Requirements

Before provisioning a TPU VM for this repo, make sure the following are true.

### 1. Correct Google account

You must be authenticated into the Google account that actually has access to the intended project and TPU quota.

Check:

```bash
gcloud config get-value account
```

Expected for the current documented setup:

```bash
ali@carezai.com
```

### 2. Correct GCP project

Check:

```bash
gcloud config get-value project
```

Expected for the current documented setup:

```bash
ir-dicompoc
```

### 3. TPU API access and permissions

Your account must be able to:

- create TPU VMs
- SSH into TPU VMs
- read TPU node metadata

### 4. GitHub access on the VM

The TPU VM needs to be able to clone:

- [bageldotcom/TPUGPU](https://github.com/bageldotcom/TPUGPU)

That means `gh` or normal Git HTTPS auth should be set up after SSH.

## Provisioning The TPU VM

The working command was:

```bash
gcloud compute tpus tpu-vm create tpugpu-v6e-1-e5a \
  --zone=us-east5-a \
  --accelerator-type=v6e-1 \
  --version=v2-alpha-tpuv6e
```

After creation, verify the node exists:

```bash
gcloud compute tpus tpu-vm describe tpugpu-v6e-1-e5a --zone=us-east5-a
```

Important fields to confirm:

- `acceleratorType: v6e-1`
- `runtimeVersion: v2-alpha-tpuv6e`
- `state: READY`

## SSH Into The TPU VM

Use:

```bash
gcloud compute tpus tpu-vm ssh tpugpu-v6e-1-e5a --zone=us-east5-a
```

Once inside, you are in a Linux VM that is attached to the TPU device. The TPU VM is conceptually:

- host VM: Linux, CPU, RAM, disk, network
- attached accelerator: TPU

## Initial Machine Inspection

Run these immediately after SSH:

```bash
python3 --version
lsb_release -a
free -h
hostname
```

What worked in the current setup:

- Python: `3.10.12`
- OS: `Ubuntu 22.04.5 LTS`
- host RAM: about `172 GiB`

At this point, `jax` is usually **not** installed yet.

## Install Required System Packages

The base image was missing some packages we needed.

Install:

```bash
sudo apt-get update
sudo apt-get install -y \
  python3.10-venv \
  python3-pip \
  git \
  gh \
  build-essential \
  tmux \
  tree \
  jq \
  unzip \
  zip
```

## Authenticate GitHub On The TPU VM

If `gh` is installed, run:

```bash
gh auth login
```

Recommended choices:

- `GitHub.com`
- `HTTPS`
- `Login with a web browser`

Verify:

```bash
gh auth status
```

## Clone The Repo On The TPU VM

Clone the Bagel repo:

```bash
git clone https://github.com/bageldotcom/TPUGPU.git
cd TPUGPU
```

Verify the remote:

```bash
git remote -v
```

Expected:

- `origin https://github.com/bageldotcom/TPUGPU.git`

## Bootstrap The Repo Environment

Do **not** install ad hoc packages manually each time. Use the repo bootstrap script:

```bash
bash scripts/setup_tpu_vm.sh
```

What the script does:

- installs required system packages
- creates a clean virtualenv at `~/tpugpu-venv`
- installs TPU JAX
- installs project Python dependencies
- installs the repo in editable mode
- runs a TPU smoke test

Important path:

- virtualenv: `~/tpugpu-venv`

Activate it later with:

```bash
source ~/tpugpu-venv/bin/activate
```

## What The Bootstrap Script Installs

At the time of writing, the script installs:

- `jax[tpu]`
- `flax`
- `optax`
- `tensorflow`
- `tensorflow-datasets`
- `orbax-checkpoint`
- `chex`
- `einops`
- `pytest`
- `rich`

It also installs the local package with:

```bash
pip install -e .
```

## TPU Verification Commands

After bootstrap, these commands should work:

```bash
source ~/tpugpu-venv/bin/activate
python -c "import jax; print(jax.__version__)"
python -c "import jax; print(jax.default_backend())"
python -c "import jax; print(jax.devices())"
```

Expected shape of output:

- JAX version prints successfully
- backend is `tpu`
- at least one `TpuDevice(...)` appears

For the documented working setup, the key result was:

- backend: `tpu`
- devices: `[TpuDevice(id=0, process_index=0, coords=(0,0,0), core_on_chip=0)]`

## Repo Workflow Rule

For this project, development should follow this loop:

1. edit code locally in the repo clone on the laptop
2. commit locally
3. push to GitHub
4. pull on the TPU VM
5. run on the TPU VM
6. inspect the result
7. repeat

Do **not** make one-off source edits directly on the TPU VM unless there is an emergency and the change will be ported back immediately.

## First Successful Training Command

This command successfully trained the first TPU-side expert after the environment was set up:

```bash
cd ~/TPUGPU
source ~/tpugpu-venv/bin/activate
python scripts/train_expert_mnist.py \
  --expert-name expert_a \
  --class-ids 0,1,2,3,4 \
  --num-epochs 1 \
  --batch-size 128 \
  --checkpoint-dir ./outputs/checkpoints \
  --seed 0
```

What this proved:

- MNIST data pipeline works on the TPU VM
- JAX/Flax training works on TPU
- the first expert can complete an epoch
- checkpoint writing works

## Known Issues We Already Resolved

These issues happened during initial bring-up and are already fixed in the repo.

### 1. MNIST tensor shape mismatch

Problem:

- the data loader assumed an extra channel dimension needed to be added
- `tensorflow_datasets` already returned MNIST with a channel dimension
- this broke image padding and resizing

Repo fix:

- `src/tpugpu/data/mnist.py` now accepts either `[N,H,W]` or `[N,H,W,C]`

### 2. Orbax checkpoint path requirement

Problem:

- Orbax required an absolute checkpoint path
- relative `./outputs/...` caused the save step to fail

Repo fix:

- `src/tpugpu/experts/train.py` now resolves checkpoint paths with `os.path.abspath(...)`

### 3. Missing TensorFlow dependency

Problem:

- `tensorflow-datasets` alone was not enough for the current data-loading path

Repo fix:

- `scripts/setup_tpu_vm.sh` now installs `tensorflow`

## Fast Re-Use Checklist

If we need to do this again on a fresh TPU VM, the minimal checklist is:

1. verify account and project
2. create TPU VM with `v6e-1` and `v2-alpha-tpuv6e`
3. SSH in
4. install system packages
5. authenticate GitHub
6. clone `bageldotcom/TPUGPU`
7. run `bash scripts/setup_tpu_vm.sh`
8. activate `~/tpugpu-venv`
9. verify `jax.default_backend()` is `tpu`
10. pull latest repo changes before every run

## Future GPU Symmetry

This repo will eventually need the same style of setup for a GPU VM.

The intended pattern is the same:

- local edit
- push to GitHub
- pull on VM
- run on VM
- repeat

When the GPU path is added, it should have its own equivalent setup document and bootstrap script rather than mixing GPU-specific instructions into the TPU bootstrap flow.
