# TPU-First DDM Guide

## Goal

This document explains how the `MNIST` DDM proof should be built if we want the first implementation to be:

- `JAX/XLA`
- `TPU-first`
- faithful to Bagel's routed DDM semantics

## Core POC Target

We want a small routed diffusion system with:

- expert A
- expert B
- router
- top-1 per-timestep routing

The first implementation should keep all of those in the `JAX` world if possible.

That gives us:

- one programming model
- one training style
- one inference style
- a cleaner path to later `TPU` and `GPU` placement

## The Exact DDM Behavior We Need

From Bagel's older `de-diffusion` code, the important routed behavior is:

1. start with noisy sample `x_t`
2. at timestep `t`, router consumes `(x_t, t)`
3. router chooses expert `e_t`
4. chosen expert predicts the denoising output
5. sampler updates the sample
6. repeat

So the core sampling primitive for this repo is:

`x_t, t, y -> router -> expert_id -> chosen expert -> x_(t-1)`

That is the central contract.

## What The Models Need To Be

### Expert

The expert should be a small class-conditional diffusion model:

- input: `x_t`
- timestep embedding: `t`
- class embedding: `y`
- output: predicted noise or velocity

For the first version, a small `UNet` is the right choice.

### Router

The router should be a small classifier:

- input: `x_t, t`
- output: `expert_id`

It does not need to be fancy.

For `MNIST`, even a small CNN plus time embedding is enough.

## Why TPU-First Changes The Design

If this were just a fast PyTorch prototype, the implementation might be more ad hoc.

Because this is `TPU-first`, we should structure it around:

- pure functions
- explicit state
- stable shapes
- compiled steps

That means we should avoid early design choices that create unnecessary dynamic behavior or hidden state.

## Recommended Build Order

### Phase A: One expert on TPU

First implement:

- data loader
- class-conditional expert model
- DDPM loss
- compiled `train_step`

The purpose is to learn the core JAX training pattern.

### Phase B: Two experts on TPU

Train:

- expert A on digits `0-4`
- expert B on digits `5-9`

Now the DDM specialization exists, but routing does not yet.

### Phase C: Router on TPU

Implement router training with:

- clean MNIST image
- sampled timestep `t`
- noised sample `x_t`
- label = expert assignment

This matches the older Bagel DDM logic closely.

### Phase D: Routed sampling on TPU

Implement top-1 sampling:

- route every step
- choose expert
- run chosen expert
- update sample

At this point, the routed DDM exists.

### Phase E: Heterogeneous placement

After the routed DDM works, move one expert to `GPU`.

The point is:

- do not debug routing and heterogeneity at the same time

## What The JAX Code Should Look Like

At a high level, each training component should be organized like this:

### Expert training

- `init_expert_model()`
- `create_expert_train_state()`
- `expert_loss_fn(params, batch, rng)`
- `expert_train_step(state, batch, rng)`

### Router training

- `init_router_model()`
- `create_router_train_state()`
- `router_loss_fn(params, batch, rng)`
- `router_train_step(state, batch, rng)`

### Sampling

- `sample_expert_step(...)`
- `route_step(...)`
- `sample_ddm_top1(...)`

The point is to keep the structure explicit and teachable.

## What The Data Should Look Like

The easiest first split is:

- expert A: `0,1,2,3,4`
- expert B: `5,6,7,8,9`

Each training batch for the expert model should contain:

- image
- class label

Each training batch for the router should contain:

- image
- class label
- expert target

The router batch is transformed at runtime into:

- `x_t`
- `t`
- expert target

## What We Should Learn At Each Stage

### While implementing the expert

Learn:

- model init/apply split
- explicit RNG handling
- compiled train steps
- train state structure

### While implementing the router

Learn:

- how auxiliary models fit into the same JAX training pattern
- how to build noisy-state supervision
- how to keep shapes stable

### While implementing the sampler

Learn:

- how to express the denoising loop cleanly
- what can stay in Python
- what should be compiled
- how routing changes the sampler semantics

### While moving to heterogeneous placement

Learn:

- how to think about device placement explicitly
- what assumptions are hardware-agnostic
- what assumptions are runtime-specific

## What To Avoid

Avoid these early mistakes:

- trying to make the first version too generic
- introducing TPU and GPU heterogeneity before routing works
- jumping to large latent diffusion models before the small model works
- copying PyTorch control flow blindly into JAX
- hiding state mutations inside too much abstraction

## Minimal Honest Success

The first honest success state is:

> a two-expert routed MNIST DDM implemented in JAX/XLA, running on TPU, with the router selecting experts at each denoising step

Everything after that is a stronger demo, not the first proof.
