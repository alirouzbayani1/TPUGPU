# Distributed Serving README

## Purpose

This document explains the distributed inference path currently implemented in `TPUGPU`.

It covers:

- the TPU expert server
- the binary wire protocol
- the router-side expert client
- the first successful cross-region sampling loop

## Current Serving Topology

Right now the distributed serving topology is:

- `TPU expert server`
  - VM: `tpugpu-v6e-1-e5a`
  - region: `us-east5-a`
  - public endpoint: `http://34.162.118.249:8000`

- `Router/orchestrator`
  - VM: `tpugpu-router-euw4a`
  - region: `europe-west4-a`

The router calls the TPU expert across regions.

## Why The Transport Is Binary

Per-step DDM routing is extremely chatty.

At every denoising step:

1. router chooses an expert
2. router sends `x_t, t, y`
3. expert returns a velocity field
4. router updates the state

If this is done with JSON, it becomes unnecessarily slow and fragile.

So the repo uses a compact binary payload based on `numpy` `.npz`.

## Wire Contract

Request payload:

- `x_t`
- `t`
- `y`

Response payload:

- `velocity`

Current protocol implementation:

- [protocol.py](/Users/ali/projects/TPUGPU/src/tpugpu/serving/protocol.py)

Unit test:

- [test_serving_protocol.py](/Users/ali/projects/TPUGPU/tests/test_serving_protocol.py)

That test is the first key transport test because if this round-trip is wrong, the distributed loop is invalid.

## TPU Expert Server

Current server script:

- [serve_expert_mnist.py](/Users/ali/projects/TPUGPU/scripts/serve_expert_mnist.py)

It:

- loads the trained expert checkpoint once
- keeps the model warm in memory
- exposes:
  - `GET /health`
  - `POST /predict`

Inference helpers:

- [inference.py](/Users/ali/projects/TPUGPU/src/tpugpu/experts/inference.py)

## Router-Side Client

Current client:

- [expert_client.py](/Users/ali/projects/TPUGPU/src/tpugpu/router/expert_client.py)

It:

- encodes `x_t, t, y`
- sends them to the remote expert endpoint
- decodes returned velocity

## First Distributed Sampler

Current remote sampling script:

- [sample_distributed_mnist.py](/Users/ali/projects/TPUGPU/scripts/sample_distributed_mnist.py)

For the first distributed proof, it:

- runs on the EU router VM
- starts from Gaussian noise
- calls the remote TPU expert at every denoising step
- writes a generated grid locally on the router VM

First successful output:

- `/home/ali/TPUGPU/outputs/distributed/tpu_only_grid.png`

Downloaded local copy:

- `/Users/ali/Desktop/TPUGPU_distributed_tpu_only/tpu_only_grid.png`

## What This Already Proves

It proves:

- the router VM can call a remote expert over the public network
- the expert can serve per-step denoising requests
- the router can hold `x_t` locally and update it step by step
- the current DDM-style distributed loop is technically viable

## Current Limitation

This first serving loop is slow because it performs:

- one remote call per denoising step
- across regions

That is not a bug. It is exactly the systems pressure this architecture creates.

Current transport optimizations are intentionally minimal.

## Obvious Next Serving Step

Once the far-region GPU expert is stable:

- expose the same server there
- point expert node B in the router to the GPU endpoint
- let the router choose between US TPU and Asia GPU per step

That is the next real serving milestone.
