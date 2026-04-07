# Live Demo README

## Purpose

This document explains the web demo now running on the EU router VM.

The goal of the demo is not to show a pretty MNIST model. The goal is to make the distributed DDM loop visible:

- the router holds the noisy state
- the noisy state updates live
- the selected expert line flashes at every inference step
- the whole system is distributed across regions

## Current Demo Topology

- `Router UI`
  - VM: `tpugpu-router-euw4a`
  - region: `europe-west4-a`
  - public URL: [http://34.141.174.148:8080](http://34.141.174.148:8080)

- `Expert node A`
  - currently points to the US TPU expert

- `Expert node B`
  - also currently points to the same US TPU expert

This is deliberate for now.

The purpose of the first UI pass is:

- distributed loop visualization
- live router-held state rendering
- animation and wiring proof

not real two-expert semantics yet.

## What The Demo Shows

The page contains:

- left expert icon
- center router icon
- right expert icon
- a live canvas above the router

During a run:

- the canvas shows the current `x_t` state
- each denoising step updates the image
- the chosen line glows
- the selected expert badge updates

## Why Both Experts Point To The Same TPU Right Now

Because the real goal for this stage was:

- get the UI live
- prove the router UI can drive the distributed inference loop
- make the zapping network effect visible

The real second expert is still finishing on the far-region A100 path.

So for now:

- both nodes use the same backend
- routing strategy is visual/demo-oriented

Supported strategies in the current UI:

- `alternating`
- `switch_halfway`
- `oracle`

## App Implementation

Main app entry:

- [run_router_demo.py](/Users/ali/projects/TPUGPU/scripts/run_router_demo.py)

Backend:

- [app.py](/Users/ali/projects/TPUGPU/src/tpugpu/demo/app.py)

Frontend:

- [index.html](/Users/ali/projects/TPUGPU/src/tpugpu/demo/static/index.html)
- [styles.css](/Users/ali/projects/TPUGPU/src/tpugpu/demo/static/styles.css)
- [app.js](/Users/ali/projects/TPUGPU/src/tpugpu/demo/static/app.js)

## Transport Between Browser And Router

The browser talks to the router app over HTTP + Server-Sent Events.

The router app then talks to the expert backend over the existing binary protocol.

So the stack is:

`browser -> EU router app -> remote expert server`

## Demo Flow

1. User opens the public page
2. User chooses a label and step count
3. User clicks `Run Live Demo`
4. Browser opens an SSE stream
5. Router app runs the denoising loop
6. Each step:
   - chooses an expert
   - calls the expert endpoint
   - updates `x_t`
   - streams the new frame to the browser
7. Browser updates the canvas and line glow

## Firewall And Exposure

The demo app required:

- the process listening on `0.0.0.0:8080`
- a GCP firewall rule opening `tcp:8080`
- a VM tag that matches that firewall rule

That is why the router demo is publicly reachable now.

## Next Demo Upgrade

When the Asia GPU expert is ready, the next change is:

- point node B to the GPU server
- use a real routing policy instead of the fake visual alternation

At that point the UI becomes a true multi-expert cross-region demo instead of a single-expert visualization.
