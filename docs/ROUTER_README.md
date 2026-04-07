# Router README

## Purpose

This document explains the router side of the `TPUGPU` POC:

- what data the router sees
- what target it is trained against
- what loss it minimizes
- where it runs
- why current router accuracy is trivial

## Current Router VM

- VM: `tpugpu-router-euw4a`
- zone: `europe-west4-a`
- machine type: `e2-standard-4`
- backend: `cpu`

The router intentionally runs in Europe so the demo is geographically distributed:

- `TPU expert` in the US
- `GPU expert` in Asia
- `Router` in Europe

## Router Input

The current router input is:

`(x_t, t, y)`

where:

- `x_t` is the noisy diffusion state
- `t` is the continuous timestep
- `y` is the class label

This matches the expert-side conditioning contract.

## Router Data Construction

The router is not trained on clean MNIST directly.

Each training example is built from:

- clean image `x_0`
- Gaussian noise `x_1`
- timestep `t ~ Uniform(0,1)`

Then:

`x_t = (1 - t) * x_1 + t * x_0`

So the router sees the same family of noisy states the experts see.

## Current Target

The current router is trained in `oracle` mode.

That means the target expert is:

- expert `0` if label is in `0..4`
- expert `1` if label is in `5..9`

This is implemented as classification over expert IDs.

## Current Loss

The router loss is:

- softmax cross-entropy over expert IDs

So mathematically:

`L_router = CE(router(x_t, t, y), target_expert_id)`

Implementation:

- [model.py](/Users/ali/projects/TPUGPU/src/tpugpu/router/model.py)
- [train.py](/Users/ali/projects/TPUGPU/src/tpugpu/router/train.py)
- [train_router_mnist.py](/Users/ali/projects/TPUGPU/scripts/train_router_mnist.py)

## Why Router Accuracy Is 100 Percent

The current router baseline reached 100% because:

- the target is a direct class partition
- the router also receives the class label `y`

So the task is easy.

This run proves:

- the router training pipeline works
- the noisy-state input pipeline works
- reporting and artifact generation work

It does not prove a strong discovered routing policy yet.

## Real Next Router Objective

The next stronger router target should be:

1. run both experts on the same `(x_t, t, y)`
2. compare each expert to the true flow-matching target
3. assign the label to the lower-error expert
4. train router on that expert-performance label

That is the true DDM-style next step.

## Router Artifacts

Current router run:

- name: `router_mnist_oracle`

Output directory on the router VM:

- `/home/ali/TPUGPU/outputs/router/router_mnist_oracle`

Artifacts include:

- per-epoch confusion matrices
- per-class routing accuracy
- predicted expert histograms
- metrics history
- router training curves
- summary JSON

## End-to-End Router Flow

The current router workflow is:

1. prepare MNIST on the EU router VM
2. construct noisy states from MNIST
3. assign oracle expert targets
4. train router classifier
5. export paper-ready artifacts

## Why The Router Still Matters Now

Even though the oracle baseline is easy, it was still useful because it established:

- the router service host
- the router training environment
- the noisy-state data pipeline
- the artifact/reporting path
- the location where the live distributed demo now runs
