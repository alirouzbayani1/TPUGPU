# TPUGPU Project Plan

## Objective

Build the smallest technically honest proof that a `DDM` can route across heterogeneous accelerators:

- one expert on `TPU`
- one expert on `GPU`
- one router making `per-timestep` expert decisions
- one logical diffusion system

The first milestone is `MNIST`, not because it is the final target, but because it is the fastest way to validate the routing architecture.

The first implementation priority is explicitly:

> `JAX/XLA on TPU first`

This is both a technical and learning decision. We want this repo to teach the `Google stack` from first principles while we build the POC.

## Scope

### In scope for v0

- `MNIST`
- `2` experts
- class-conditional diffusion
- expert specialization on disjoint digit groups
- separately trained router on noisy inputs
- top-1 routed denoising
- one-stack implementation first
- then heterogenous placement

### Out of scope for v0

- Paris 2
- Open-Sora
- video
- latent diffusion
- text conditioning
- learned multi-expert blending
- production deployment

## Milestones

### M0: TPU-first JAX routed DDM on MNIST

Goal:

- prove the DDM mechanics in the `JAX/XLA` stack before introducing heterogeneous placement

Success criteria:

- one small expert trains successfully on `TPU`
- a second expert can be trained with the same `JAX` code path
- router predicts expert IDs from noisy states
- routed top-1 sampler produces sensible results
- forced wrong-expert behavior is visibly worse than routed behavior

### M1: TPU/GPU heterogeneous DDM on MNIST

Goal:

- keep the exact same `JAX` model family and move one expert to `GPU` while the other stays on `TPU`

Success criteria:

- same routed sampler works with cross-device experts
- both experts are callable through one logical sampling path
- sample generation completes end to end

### M2: Stronger external demo

Goal:

- move from a toy proof to a stronger, more externally legible proof

Likely candidates:

- `CIFAR-10`
- later, a shared real model family such as `Stable Diffusion 2.x`

## Technical Design

### Dataset split

First-pass split:

- Expert A: digits `0,1,2,3,4`
- Expert B: digits `5,6,7,8,9`

Rationale:

- clear specialization
- easy evaluation
- simple router labels

Known limitation:

- the router may become mostly static because the split is very clean

That is acceptable for `v0`.

### Expert model

Use a small `class-conditional DDPM` with a `UNet` backbone.

Implementation priority:

- `JAX` model definition
- `XLA`-compiled training step
- TPU as the first target runtime

Expert interface:

- input: `x_t, t, y`
- output: denoising prediction

### Router model

Use a small classifier over noisy states.

Router interface:

- input: `x_t, t`
- output: `expert_id`

The router should follow the same high-level behavior as Bagel's older `de-diffusion` repo:

- it runs during denoising
- it selects the expert at each timestep
- only the selected expert denoises that step

### Sampler

Sampling loop for `top1`:

1. initialize noise `x_T`
2. for each timestep `t`
3. route `x_t` through the router
4. choose expert `e_t`
5. run only expert `e_t`
6. update sample to the next timestep

This is the core behavior that makes the system a real routed DDM rather than a front-door ensemble.

## Implementation Steps

### Step 0: Learn the stack while building

This project is intentionally also a learning project.

The first docs to study are:

- `/Users/ali/projects/TPUGPU/docs/JAX_XLA_PRIMER.md`
- `/Users/ali/projects/TPUGPU/docs/TPU_FIRST_DDM_GUIDE.md`

### Step 1: Data layer

Deliverables:

- MNIST loader
- expert split definitions
- filtered training subsets per expert
- optional helper to create router labels

### Step 2: Expert training

Deliverables:

- small class-conditional UNet in `JAX`
- DDPM training loop in `JAX`
- compiled `train_step`
- TPU-targeted training state
- config for expert A
- config for expert B
- checkpoint saving/loading

### Step 3: Router training

Deliverables:

- noisy-state router model
- noise injection utility for router training
- cross-entropy training loop to predict expert ID
- checkpoint saving/loading

### Step 4: Routed sampling

Deliverables:

- DDPM scheduler utilities
- top-1 router-in-the-loop sampling
- ablation modes:
  - forced expert A
  - forced expert B
  - routed

### Step 5: Evaluation

Deliverables:

- sample grids
- per-expert sample comparison
- wrong-expert ablation
- basic router accuracy metrics

### Step 6: Heterogeneous placement

Deliverables:

- device-aware expert wrappers
- one expert on `GPU`
- one expert on `TPU`
- same routed sampling path

## Repo Layout

Planned structure:

- `src/tpugpu/data`
- `src/tpugpu/experts`
- `src/tpugpu/router`
- `src/tpugpu/sampling`
- `src/tpugpu/eval`
- `scripts`
- `tests`

## Immediate Build Order

Build in this order:

1. data split
2. single expert training in `JAX/XLA`
3. second expert training in `JAX/XLA`
4. router training in `JAX/XLA`
5. routed sampler in `JAX/XLA`
6. evaluation script
7. TPU/GPU split

Do not move to TPU until routed MNIST works on one stack.

Revision:

For this repo, `one stack first` means `JAX/XLA on TPU`.

## Risks

### Risk 1: Router degenerates into static domain selection

Expected for the first split.

Mitigation:

- accept it for `v0`
- later introduce overlapping competence or timestep specialization

### Risk 2: TPU integration obscures routing bugs

Mitigation:

- complete `M0` before `M1`

### Risk 3: Framework mismatch between GPU and TPU

Mitigation:

- first keep the implementation as unified as possible
- only introduce mixed-runtime complexity after the model and router are stable

## First Concrete Deliverable

The first concrete deliverable is:

> a routed MNIST DDM with two experts and a separately trained router, implemented in `JAX/XLA` and running correctly on TPU

Once that exists, moving one expert to TPU becomes an engineering task instead of a research task.

Correction to the original wording:

After this revision, moving one expert to GPU becomes an engineering task instead of a research task.
