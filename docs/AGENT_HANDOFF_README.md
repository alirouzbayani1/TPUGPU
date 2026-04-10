# Agent Handoff README

This document is the single-source handoff for another coding agent to recreate, operate, debug, and extend the current `TPUGPU` distributed MNIST DDM proof-of-concept.

It is intentionally redundant. The point is to remove guesswork.

It explains:

- what the system is
- which machines exist
- what is running where
- which checkpoints and artifacts matter
- how the router, TPU expert, and GPU expert fit together
- how to restart the live demo and expert servers
- how to retrain the router and experts
- what bugs and operational traps already happened

If this project needs to be rebuilt from scratch by another agent, this is the first file to read.

## 1. What This System Is

This repo currently demonstrates a small, real, geographically distributed DDM-style system using MNIST instead of text-to-image.

The analogy is:

- router holds the current noisy state
- at each denoising step, router selects an expert
- router sends `(x_t, t, y)` to the chosen expert
- expert returns a predicted velocity
- router updates the state and repeats

The current conditioning variable is:

- `y` = MNIST digit class label

not text.

The architectural point is the same as DDM:

- multiple remote experts
- lightweight router
- expert choice at each denoising step
- cross-region inference loop

## 2. Current Machine Topology

There are three important live machines.

### 2.1 TPU expert

- VM name: `tpugpu-v6e-1-e5a`
- zone: `us-east5-a`
- accelerator: `v6e-1`
- role: expert A
- training split: digits `0,1,2,3,4`
- expert name: `expert_strict_0_4_fm`
- public expert endpoint: `http://34.162.118.249:8000`

### 2.2 GPU expert

- VM name: `tpugpu-gpu-a100-sg1a`
- zone: `asia-southeast1-a`
- accelerator: `NVIDIA A100 40GB`
- role: expert B
- training split: digits `5,6,7,8,9`
- expert name: `expert_strict_5_9_gpu_fm_a100`
- public expert endpoint: `http://34.143.140.34:8000`

### 2.3 Router + UI

- VM name: `tpugpu-router-euw4a`
- zone: `europe-west4-a`
- machine type: `e2-standard-4`
- role: router training host, orchestration host, UI host
- public UI: `http://34.141.174.148:8080`
- public health endpoint: `http://34.141.174.148:8080/api/health`

This gives the current three-region story:

- US TPU expert
- Europe router
- Asia GPU expert

## 3. What The Experts Actually Are

Both experts use the same core modeling interface:

- input: `(x_t, t, y)`
- output: `velocity`

The current expert model is:

- dataset: `MNIST`
- architecture: class-conditional small UNet
- objective: `flow matching`
- framework: JAX-based implementation in this repo

Important distinction:

- the UI card labels the GPU as `PyTorch / CUDA` for the conceptual TPU-vs-GPU stack comparison
- the current actual GPU implementation in this repo is still JAX-based

So:

- conceptual stack in demo copy: TPU = `JAX / XLA`, GPU = `PyTorch / CUDA`
- actual implementation in code: both expert code paths are in this JAX codebase

Do not confuse those two statements.

## 4. Flow-Matching Objective

Each expert trains on:

- clean image `x_0`
- Gaussian noise `x_1`
- timestep `t ~ Uniform(0, 1)`

Noisy state:

`x_t = (1 - t) * x_1 + t * x_0`

Target velocity:

`v* = x_0 - x_1`

The expert learns:

`f(x_t, t, y) -> v`

Relevant files:

- [model.py](/Users/ali/projects/TPUGPU/src/tpugpu/experts/model.py)
- [train.py](/Users/ali/projects/TPUGPU/src/tpugpu/experts/train.py)
- [train_expert_mnist.py](/Users/ali/projects/TPUGPU/scripts/train_expert_mnist.py)

## 5. Strict Split Design

The current real split is:

- expert A: digits `0..4`
- expert B: digits `5..9`

This is a strict label split.

That means:

- TPU expert is supposed to be good on `0,1,2,3,4`
- GPU expert is supposed to be good on `5,6,7,8,9`

Held-out behavior was explicitly tested:

- strict `0..4` experts forced to generate `5..9`
- strict `5..9` expert forced to generate `0..4`

Those outputs were saved locally and demonstrate specialization failure on held-out labels.

## 6. Router Design

The current router is trained as a classifier over expert IDs.

Router input:

`(x_t, t, y)`

Current training target:

- expert `0` if `y in 0..4`
- expert `1` if `y in 5..9`

Current router loss:

- softmax cross-entropy over expert IDs

Current router run:

- name: `router_mnist_oracle`

Important caveat:

This is still the easy/oracle router, not the future stronger router based on measured expert competence.

The next stronger router objective would be:

1. run both experts on the same `(x_t, t, y)`
2. compare each to the true flow target
3. label the better expert
4. train router on that

That is not the current training target.

Relevant files:

- [model.py](/Users/ali/projects/TPUGPU/src/tpugpu/router/model.py)
- [train.py](/Users/ali/projects/TPUGPU/src/tpugpu/router/train.py)
- [train_router_mnist.py](/Users/ali/projects/TPUGPU/scripts/train_router_mnist.py)
- [inference.py](/Users/ali/projects/TPUGPU/src/tpugpu/router/inference.py)

## 7. Router Checkpoint Status

This matters because there was a bug here before.

Originally:

- router training ran
- plots and metrics were produced
- but no loadable router checkpoint was saved

That was fixed.

Current state:

- router training now writes checkpoints
- live demo can load the checkpoint

Expected checkpoint location on the EU router VM:

- `/home/ali/TPUGPU/outputs/router_checkpoints/router_mnist_oracle/step_20`

If the live demo health says:

- `{"status":"ok","router_loaded":"yes"}`

then the router checkpoint was loaded successfully by the live app.

## 8. Distributed Serving Contract

Each expert server exposes:

- `GET /health`
- `POST /predict`

The router sends:

- `x_t`
- `t`
- `y`

The expert returns:

- `velocity`

The transport is binary and uses the helper in:

- [protocol.py](/Users/ali/projects/TPUGPU/src/tpugpu/serving/protocol.py)

Client:

- [expert_client.py](/Users/ali/projects/TPUGPU/src/tpugpu/router/expert_client.py)

Server:

- [serve_expert_mnist.py](/Users/ali/projects/TPUGPU/scripts/serve_expert_mnist.py)

Expert inference helper:

- [inference.py](/Users/ali/projects/TPUGPU/src/tpugpu/experts/inference.py)

## 9. Live Demo Status

The live demo UI is served from the EU router VM.

Important backend truth:

- expert `0` = real TPU backend
- expert `1` = real A100 GPU backend
- router selection in the live demo is now driven by the trained router checkpoint, not the old staged alternating logic

Important frontend truth:

- the browser still passes a `strategy` param in the query string
- but the actual routing path on the backend uses the learned router when a router checkpoint is loaded

The seed bug was fixed:

- each `Generate` click now sends a fresh random seed
- server also supports generating a fresh seed if missing

So repeated runs with the same label and same step count should no longer start from identical noise.

Relevant files:

- [run_router_demo.py](/Users/ali/projects/TPUGPU/scripts/run_router_demo.py)
- [app.py](/Users/ali/projects/TPUGPU/src/tpugpu/demo/app.py)
- [index.html](/Users/ali/projects/TPUGPU/src/tpugpu/demo/static/index.html)
- [styles.css](/Users/ali/projects/TPUGPU/src/tpugpu/demo/static/styles.css)
- [app.js](/Users/ali/projects/TPUGPU/src/tpugpu/demo/static/app.js)

## 10. Exact Local Workflow Convention

This has been important throughout the project.

Expected workflow:

1. edit locally on the laptop in `/Users/ali/projects/TPUGPU`
2. commit locally
3. push to GitHub
4. pull on the relevant VM
5. run/restart on that VM

Do not silently patch code only on a VM unless there is a temporary emergency.

The user explicitly wanted:

- local edit first
- push
- pull on VM
- repeat

## 11. Current Important Artifact Locations

### TPU strict `0..4`

Local:

- `/Users/ali/Desktop/TPUGPU_tpu_strict_0_4`
- `/Users/ali/Desktop/TPUGPU_tpu_strict_0_4_heldout`

VM:

- `/home/ali/TPUGPU/outputs/checkpoints/expert_strict_0_4_fm`

### GPU strict `5..9`

Local:

- `/Users/ali/Desktop/TPUGPU_gpu_strict_5_9_a100`
- `/Users/ali/Desktop/TPUGPU_gpu_strict_5_9_heldout`

VM:

- `/home/ali/TPUGPU/outputs/checkpoints/expert_strict_5_9_gpu_fm_a100`

### Router

VM:

- `/home/ali/TPUGPU/outputs/router/router_mnist_oracle`
- `/home/ali/TPUGPU/outputs/router_checkpoints/router_mnist_oracle`

## 12. Current Final GPU Expert Outcome

The far-region A100 expert finished.

Final observed outcome:

- reached `epoch 50`
- final train loss: `0.118632`
- final PCA-FID proxy: `50.953739`

Checkpoint exists:

- `/home/ali/TPUGPU/outputs/checkpoints/expert_strict_5_9_gpu_fm_a100`

Training is not currently running there anymore.

## 13. Known Operational Problems

### 13.1 `gcloud ssh` intermittently fails with code `255`

This was a repeated operational problem.

Symptoms:

- `gcloud compute ssh ... --command="...long shell chain..."` sometimes fails
- especially on restart sequences
- the code may already have been pulled, but the process restart part can fail

Implication:

- do not assume “pull + restart” fully happened just because part of the command ran
- check service health afterward

### 13.2 EU demo server restart is fragile

The most reliable restart pattern turned out to be the simpler one:

- separate or short start commands
- not overly complicated shell chains

### 13.3 Expert servers can be down independently

When the browser shows:

- `ERR_INCOMPLETE_CHUNKED_ENCODING`

that often means:

- router stream started
- selected expert backend was unreachable
- router crashed mid-stream because the remote expert call failed

This happened before with:

- GPU server down for label `9`
- TPU server down for label `2`

So when debugging stream failures:

1. check router health
2. check TPU expert health
3. check GPU expert health

## 14. Exact Health Checks

Router:

```bash
curl -s http://34.141.174.148:8080/api/health
```

TPU expert:

```bash
curl -s http://34.162.118.249:8000/health
```

GPU expert:

```bash
curl -s http://34.143.140.34:8000/health
```

Expected healthy router response:

```json
{"status":"ok","router_loaded":"yes"}
```

Expected healthy expert response shape:

```json
{"status":"ok","expert_name":"..."}
```

## 15. Reliable-ish Restart Commands

These are the patterns that have been used repeatedly.

### 15.1 Restart router UI

```bash
gcloud compute ssh tpugpu-router-euw4a --zone=europe-west4-a \
  --command="bash -lc 'cd /home/ali/TPUGPU && nohup /home/ali/tpugpu-router-venv/bin/python scripts/run_router_demo.py --host 0.0.0.0 --port 8080 >/home/ali/TPUGPU/outputs/demo.log 2>&1 < /dev/null & echo STARTED:\$!'"
```

### 15.2 Restart TPU expert server

```bash
gcloud compute tpus tpu-vm ssh tpugpu-v6e-1-e5a --zone=us-east5-a \
  --command="bash -lc 'cd /home/ali/TPUGPU && source /home/ali/tpugpu-venv/bin/activate && nohup python scripts/serve_expert_mnist.py --expert-name expert_strict_0_4_fm --batch-size 64 --port 8000 >/home/ali/TPUGPU/outputs/expert_server_tpu.log 2>&1 < /dev/null & echo STARTED:\$!'"
```

### 15.3 Restart GPU expert server

```bash
gcloud compute ssh tpugpu-gpu-a100-sg1a --zone=asia-southeast1-a \
  --command="bash -lc 'cd /home/ali/TPUGPU && nohup /home/ali/tpugpu-gpu-venv/bin/python scripts/serve_expert_mnist.py --expert-name expert_strict_5_9_gpu_fm_a100 --batch-size 64 --port 8000 >/home/ali/TPUGPU/outputs/expert_server_gpu.log 2>&1 < /dev/null & echo STARTED:\$!'"
```

If these fail because of SSH transport, retry with shorter commands rather than adding more shell complexity.

## 16. Logging Already Added For Debugging

The demo backend logs:

- `demo_start`
- `demo_step`
- `demo_velocity`
- `demo_done`

The expert servers log:

- `expert_predict`
- `expert_velocity`

Those logs were added specifically to prove:

- which expert was selected
- which endpoint was called
- which label and timestep were used
- whether the returned velocity looked sane

Relevant files:

- [app.py](/Users/ali/projects/TPUGPU/src/tpugpu/demo/app.py)
- [serve_expert_mnist.py](/Users/ali/projects/TPUGPU/scripts/serve_expert_mnist.py)

## 17. UI State

The UI has gone through several redesigns.

Current stable design choices:

- world map in one SVG coordinate system
- pins, lines, labels, and cards all inside that SVG
- no more mixed coordinate-system hack from image + overlays
- cards now auto-size to content
- cards align to the same text columns as titles/regions
- route lines flash neon green on selection
- router-held noisy state updates live in the central black square

Important lesson:

The old bug came from mixing:

- a map image
- pins in one coordinate system
- lines in another

That approach was wrong. The SVG rewrite was the correct fix.

## 18. Things Another Agent Should Not Re-Break

Do not regress these:

1. router checkpoint loading
- router was trained before but not saved
- that bug is fixed now

2. random seed per UI run
- stale-client caching once made runs deterministic
- cache-busting and fresh seed behavior were added

3. actual expert mapping
- expert 0 must remain TPU
- expert 1 must remain GPU

4. one-SVG UI layout
- do not go back to HTML image + absolute-position overlays

5. local-first workflow
- edit local
- push
- pull on VM

## 19. Most Likely Next Technical Expansion

The current router is still an oracle router.

The next meaningful expansion is:

- replace oracle labels with expert-performance labels

That means:

1. generate `(x_t, t, y)`
2. run both experts
3. compare each to the true flow target
4. assign the lower-error expert
5. retrain the router

That is the clean next step if the goal is to make the router less trivial and more genuinely DDM-like.

## 20. If You Need To Rebuild From Scratch

Minimal rebuild order:

1. clone repo locally
2. set up TPU VM and GPU VM and router VM
3. train strict TPU expert on `0..4`
4. train strict GPU expert on `5..9`
5. train router on EU CPU VM with checkpoint saving
6. start TPU expert server
7. start GPU expert server
8. start router UI in Europe
9. verify all three public health endpoints
10. test labels `2` and `9` through the live UI

If either label fails:

- label `2` should hit TPU
- label `9` should hit GPU

That is the fastest sanity check for end-to-end routing.

## 21. Related Docs

This file is the unified handoff.

The older split docs still exist:

- [EXPERTS_README.md](/Users/ali/projects/TPUGPU/docs/EXPERTS_README.md)
- [ROUTER_README.md](/Users/ali/projects/TPUGPU/docs/ROUTER_README.md)
- [SERVING_README.md](/Users/ali/projects/TPUGPU/docs/SERVING_README.md)
- [LIVE_DEMO_README.md](/Users/ali/projects/TPUGPU/docs/LIVE_DEMO_README.md)

Those are still useful, but this document should be enough for a new agent to take over the project without prior chat history.
