# JAX and XLA Primer

## Why This Document Exists

This repo is not only a POC repo. It is also a learning repo.

The goal is that by the time the `TPUGPU` POC works, the engineer building it should understand:

- what `JAX` is
- what `XLA` is
- how `TPU`-oriented ML code is usually structured
- how this differs from a normal `PyTorch + CUDA` workflow

This document explains those ideas from first principles.

## The Short Version

`JAX` is the programming model.  
`XLA` is the compiler.  
`TPU` is one hardware target that `XLA` can compile to.

So the common stack is:

`Python -> JAX program -> XLA compile -> TPU execution`

## Mental Model

### PyTorch world

In the usual PyTorch world, you often think like this:

- write imperative tensor code
- tensors live on devices
- execute ops one by one
- CUDA libraries and kernels do the heavy lifting
- optimize by changing kernels, precision, memory use, and distributed strategy

### JAX world

In JAX, the more accurate mental model is:

- write pure numerical functions
- tell JAX which functions should be transformed
- let XLA compile whole function graphs into optimized execution

That means JAX code is often less about step-by-step imperative execution and more about describing computation in a way the compiler can understand and optimize.

## What JAX Actually Gives You

The core building blocks are:

- `jax.numpy`
  - NumPy-like array programming
- `jax.grad`
  - automatic differentiation
- `jax.jit`
  - compile a function with XLA
- `jax.vmap`
  - vectorize a function over a batch dimension
- `jax.pmap` / `pjit`
  - distribute computation across devices

If you understand those, you understand most of the conceptual surface.

## What XLA Actually Does

`XLA` stands for `Accelerated Linear Algebra`, but the important thing is not the name. The important thing is the role:

> XLA takes a high-level array computation and compiles it into an optimized executable for hardware such as TPU or GPU.

That changes how you should think about performance.

In a JAX/XLA stack, performance depends heavily on:

- whether the computation can be compiled as one optimized graph
- whether shapes stay stable across steps
- whether device communication and data movement are predictable

This is why JAX/XLA code often cares so much about:

- static shapes
- explicit batch sizes
- explicit device meshes
- explicit sharding

## Why TPU Code Feels Different

On TPU, the common workflow is more compiler-centric and system-centric than in the mainstream CUDA world.

TPU-friendly code usually tries to be:

- shape-stable
- large-grained
- compiler-friendly
- explicit about distribution

That means less emphasis on custom hand-written kernels and more emphasis on making the whole training step compile and shard well.

## Pure Functions Matter

One of the most important JAX concepts is:

> JAX transformations work best on pure functions.

A pure function:

- depends only on its inputs
- produces outputs without mutating hidden global state

This matters because `jit`, `grad`, and device transformations all assume the computation can be reasoned about functionally.

This is different from the common PyTorch habit of writing logic that mutates module state freely during execution.

## Arrays and Immutability

In JAX, arrays are conceptually immutable.

That does not mean computation is slow. It means you write updates functionally.

So instead of thinking:

- mutate tensor in place

you more often think:

- return a new array representing the updated value

This is one of the mindset shifts that matters early.

## Randomness Is Explicit

In PyTorch, randomness is often implicit through global RNG state.

In JAX, randomness is explicit.

You pass and split `PRNGKey`s.

This matters because the training step often looks like:

- current state
- batch
- rng key
- returns new state, metrics, and new rng key

That explicitness is annoying at first, but it makes compiled and distributed execution easier to reason about.

## The Core ML Engineer Objects In JAX

For this project, the most important objects to understand are:

### 1. Parameters

These are the model weights.

### 2. Model function

This is the forward computation:

- input `x_t`
- timestep `t`
- class label `y`
- output predicted noise or velocity

### 3. Loss function

This compares the model prediction to the target and returns a scalar loss.

### 4. Optimizer state

This holds momentum and related optimizer bookkeeping.

### 5. Train state

In JAX projects, it is common to package:

- params
- optimizer state
- step counter
- sometimes RNG or EMA state

into a single structured object called a `train state`.

### 6. Compiled train step

This is usually the most important function:

- takes state, batch, rng
- computes loss and gradients
- applies optimizer update
- returns new state and metrics

This function is often `jit`-compiled and becomes the core high-performance training primitive.

## What a JAX Training Loop Usually Looks Like

At a high level:

1. initialize model parameters
2. initialize optimizer
3. create train state
4. iterate over batches
5. call compiled `train_step`
6. record metrics
7. checkpoint state

The key difference from ordinary Python loops is that the heavy computation is usually inside the compiled function, not spread across many ad hoc Python operations.

## What Static Shapes Mean

One of the most important practical ideas in XLA is:

> shape changes can trigger recompilation

This matters because recompilation is expensive.

So for good performance, you generally want:

- fixed image size
- fixed batch size
- fixed tensor ranks
- predictable control flow

For this repo, that is one reason `MNIST` is such a good first target.

It is small and shape-stable.

## What Sharding Means

When you move beyond one device, you need to decide how tensors are split.

That is `sharding`.

Examples:

- split batch across devices
- split model parameters across devices
- replicate some arrays everywhere

In Google-style JAX systems, sharding is a first-class design decision, not just a hidden runtime detail.

For this repo, the earliest phase does not need aggressive sharding complexity. But you should still know the concept because it is central to why JAX and TPU systems scale the way they do.

## Why JAX/XLA Is Strategically Relevant Here

This repo exists partly because the `Google` story is stronger when the workload is grounded in:

- `TPU`
- `JAX`
- `XLA`

That means the learning goal is not optional. It is part of the strategic value of the project.

If we can express the DDM cleanly in JAX/XLA, then:

- the workload becomes more naturally Google-native
- the TPU story becomes stronger
- later movement between `TPU` and `GPU` can still happen from a common model definition

## Mapping From PyTorch Terms To JAX Terms

Rough translation:

- `torch.Tensor` -> `jax.Array`
- `nn.Module` -> Flax module or pure apply/init functions
- `forward()` -> apply function
- `optimizer.step()` -> optimizer update inside compiled train step
- autograd -> `jax.grad`
- AMP/autocast -> dtype policy and compilation choices
- DDP / FSDP -> `pmap`, `pjit`, mesh/sharding
- implicit RNG -> explicit PRNG keys

The mapping is not exact, but this is the right first approximation.

## What We Need For This Repo Specifically

For the MNIST DDM proof, the JAX/XLA concepts that matter most are:

- pure train step
- explicit RNG handling
- compiled loss/gradient/update step
- stable shapes
- device placement
- later, basic sharding awareness

You do not need to master all of JAX before starting.

You need to understand enough to implement:

- expert model
- router model
- training loop
- routed denoising loop

## First Principles Summary

If you remember only five things, remember these:

1. `JAX` is a way to express array programs as transformable pure functions.
2. `XLA` compiles those functions into optimized hardware execution.
3. `TPU` performance depends heavily on shape stability and compiler-friendly structure.
4. JAX training code is usually organized around a compiled `train_step`.
5. For this repo, the right mental model is not “port PyTorch habits,” but “design the DDM so JAX/XLA can own the core loop cleanly.”
