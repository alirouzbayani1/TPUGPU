# TPUGPU

## Purpose

`TPUGPU` is a minimal proof-of-concept repo for one narrow claim:

> A Decentralized Diffusion Model (`DDM`) can operate as one logical system while placing different experts on different accelerator types, specifically `TPU` and `GPU`.

This repo is not Paris 2, not Open-Sora, and not a productization effort. It is a clean technical POC intended to validate the smallest version of the idea with minimal engineering overhead.

## Operating Docs

Use these documents as the source of truth for project execution:

- [PROJECT_PLAN.md](/Users/ali/projects/TPUGPU/PROJECT_PLAN.md)
- [docs/TPU_VM_SETUP.md](/Users/ali/projects/TPUGPU/docs/TPU_VM_SETUP.md)
- [docs/TPU_FIRST_DDM_GUIDE.md](/Users/ali/projects/TPUGPU/docs/TPU_FIRST_DDM_GUIDE.md)

`docs/TPU_VM_SETUP.md` is the exact bring-up guide for reproducing the working TPU VM state that already ran the first MNIST expert successfully.

## Why This Repo Exists

Bagel's long-term thesis is that decentralized training and decentralized inference can become a meaningful AI workload class. For Google specifically, the strategic value is strongest if that workload can become `TPU`-relevant instead of defaulting entirely to `CUDA`.

The problem with trying to prove that directly using `paris2` is that the current Paris 2 training stack is heavily tied to:

- `PyTorch`
- `CUDA`
- `NCCL`
- `ColossalAI`
- `Open-Sora`
- CUDA-oriented kernels like `flash-attn`

That makes Paris 2 a poor first vehicle for a small TPU/GPU validation.

So this repo exists to isolate the core systems question:

> Can we show a real DDM-style routed diffusion system where one expert lives on TPU, another lives on GPU, and the router composes them at inference time?

## Exact POC Goal

The first milestone is a `small MNIST DDM proof`.

The purpose of the MNIST proof is not to impress anybody visually. Its purpose is to establish the architecture quickly and honestly:

- independent experts
- expert specialization on disjoint data
- a router that selects experts during the denoising trajectory
- one expert running on `TPU`
- one expert running on `GPU`

If that works, the next phase can move to a more meaningful dataset or model family.

## What Counts As Success

The POC succeeds if all of the following are true:

1. Two experts are trained independently on disjoint data subsets.
2. One expert runs on `TPU`, one expert runs on `GPU`.
3. A router chooses which expert to use during sampling.
4. The full system can generate samples through one logical inference path.
5. The combined system behaves like a routed DDM, not just a static ensemble.

## What This Repo Is Not Trying To Prove

This repo is not trying to prove:

- that TPUs are universally better than GPUs
- that Paris 2 runs on TPU today
- that heterogeneous training is already production-ready
- that this first POC should use video models
- that a toy result is enough for the final Google story

This repo is only trying to establish a technically clean first proof.

## Current Technical Decision

### Phase 0 target

Use `MNIST` first.

Reason:

- fastest path to validate the architecture
- easiest to train from scratch
- easy to inspect failure modes
- small enough to move quickly

### Model family

Use a `small class-conditional diffusion model` with a `UNet` backbone.

Reason:

- much smaller than latent diffusion
- easier to implement on both `GPU` and `TPU`
- easy to specialize by data split
- good enough to prove routed denoising behavior

### Routing requirement

The router must behave like real DDM routing, not a one-shot gateway.

That means:

- the router should run inside the denoising loop
- it should select an expert at each timestep
- expert selection should depend on the current noisy state `x_t` and timestep `t`

This requirement comes directly from Bagel's earlier DDM codebase.

## Critical Finding From `de-diffusion`

We studied Bagel's earlier DDM repo at:

- `/Users/ali/projects/de-diffusion`

The key files were:

- `/Users/ali/projects/de-diffusion/src/inference.py`
- `/Users/ali/projects/de-diffusion/src/models.py`
- `/Users/ali/projects/de-diffusion/scripts/train_router_standalone.py`

The important takeaway is:

- DDM routing in that repo is `per-timestep`
- the router takes the current noisy sample `x_t` and timestep `t`
- in `top1` mode, the router chooses one expert per sample per timestep
- only that expert predicts the denoising update for that step

So the real target behavior is:

`router(x_t, t) -> expert_id -> chosen expert denoises this step`

This matters because it means a simple front-door prompt router is not enough if we want this POC to count as a real DDM proof.

## Important Consequence For MNIST

Pure unconditional diffusion is a weak fit for this demo because unconditional sampling gives the router no explicit task-side input beyond the noisy state.

So the current plan is:

- use `class-conditional` diffusion
- train experts on disjoint class groups
- route during denoising

This allows:

- specialization
- controlled ablations
- wrong-expert tests

Example split:

- Expert 0: digits `0-4`
- Expert 1: digits `5-9`

This is acceptable for a first proof, but it has one known limitation:

- if the data split is too clean, the router may collapse into a mostly static domain selector rather than switching meaningfully across timesteps

That is fine for `v0`. If needed later, we can induce more dynamic routing by making expert competence overlap more or by giving experts different timestep/noise specializations.

## Why We Are Not Starting With MaxDiffusion

We also studied Google's JAX diffusion reference repo at:

- `/Users/ali/projects/maxdiffusion`

MaxDiffusion is highly relevant for the later Google-facing phase, but it is not the right tool for the very first MNIST proof.

Reasons:

- it is built for real latent diffusion and video workloads
- the smallest documented trainable families there are still much larger than what we need for `MNIST`
- using it immediately would add irrelevant complexity to the first architecture proof

For the later, more realistic phase, the best overlap between Google's stack and the mainstream GPU world is likely:

- `Stable Diffusion 2.x`

That will matter after the MNIST proof works.

## Why We Are Not Starting With Paris 2

We also studied:

- `/Users/ali/projects/paris2`

Key conclusion:

Paris 2 today is best thought of as:

- Bagel expert orchestration and routing logic
- on top of an Open-Sora / CUDA / PyTorch video training core

So Paris 2 is too entangled with the current stack to serve as the first clean TPU/GPU DDM proof.

This repo should remain separate and deliberately small.

## Intended Architecture

The clean target architecture is:

- `Expert A` on `GPU`
- `Expert B` on `TPU`
- one shared diffusion model family
- one router used during denoising
- one logical inference entrypoint

Conceptually:

`x_t, t, condition -> router -> expert selection -> chosen expert predicts update -> next step`

This is closer to Bagel's original DDM semantics than a simple expert ensemble.

## First Implementation Shape

Planned components:

- `experts/`
  - training and inference code for the small class-conditional diffusion experts
- `router/`
  - training and inference code for a noisy-state router
- `sampling/`
  - DDM sampling loop with per-timestep expert selection
- `data/`
  - dataset split logic for expert-specialized training
- `eval/`
  - basic checks and visual outputs

The repo should stay minimal until the first proof works.

## Phase Plan

### Phase 0

Goal:

- get a tiny DDM working on one hardware type first if needed

Deliverables:

- small class-conditional MNIST diffusion model
- two experts trained on disjoint data
- router trained on noisy states
- routed top-1 denoising loop

### Phase 1

Goal:

- place one expert on `GPU` and one on `TPU`

Deliverables:

- same logical DDM
- heterogeneous expert placement
- end-to-end sampling through one routed path

### Phase 2

Goal:

- move from the toy proof to a more externally meaningful proof

Likely direction:

- `CIFAR-10`, then
- a real shared model family such as `Stable Diffusion 2.x`

## Evaluation Ideas

Basic `v0` evaluation should be simple and direct:

- per-expert sample grids
- routed sample grids
- wrong-expert ablation
- router accuracy on noisy states
- whether routed sampling improves behavior over forcing the wrong expert

The first success bar is correctness, not benchmark quality.

## Constraints

- Minimal engineering effort is preferred.
- The first proof should optimize for speed of validation, not model quality.
- We should avoid pulling in large framework complexity until the routing behavior is established.
- The architecture should remain faithful enough to Bagel's original DDM semantics to be intellectually honest.

## Open Questions

These are still open:

- whether the first implementation should use `JAX` on both devices or a mixed `JAX TPU + PyTorch GPU` stack
- how much dynamic per-timestep routing we can induce on `MNIST`
- whether the first router should be trained directly or partially bootstrapped from oracle labels
- what checkpoint/export contract we want once we move beyond the first toy proof

## Recommended Next Steps

1. Lock the exact MNIST split and conditioning scheme.
2. Pick the framework strategy for `TPU` and `GPU`.
3. Implement the smallest class-conditional expert model.
4. Implement router training on noisy states.
5. Implement top-1 routed sampling.
6. Move one expert to TPU once the single-stack version works.

## State Of Understanding At Repo Creation

As of `2026-04-06`, the key decisions already made are:

- start with `MNIST`
- use a `small class-conditional diffusion model`
- preserve `per-timestep routed denoising`
- keep this repo separate from `paris2`
- treat this as the first architecture proof before moving to larger Google-facing models

This README is intentionally detailed so a future session can resume without reconstructing the entire design discussion.

## Additional Project Docs

The execution plan for the first implementation pass lives in:

- [PROJECT_PLAN.md](/Users/ali/projects/TPUGPU/PROJECT_PLAN.md)

The learning and stack notes live in:

- [docs/TPU_FIRST_DDM_GUIDE.md](/Users/ali/projects/TPUGPU/docs/TPU_FIRST_DDM_GUIDE.md)
