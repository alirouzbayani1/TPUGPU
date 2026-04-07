from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from tpugpu.router.expert_client import ExpertClient

STATIC_DIR = Path(__file__).resolve().parent / "static"


def _normalize_frame(x_t: np.ndarray) -> list[int]:
    image = np.asarray(x_t[0, :, :, 0], dtype=np.float32)
    image = np.clip((image + 1.0) * 127.5, 0.0, 255.0).astype(np.uint8)
    return image.reshape(-1).tolist()


def _select_expert(step_idx: int, total_steps: int, label: int, strategy: str) -> int:
    if strategy == "alternating":
        return step_idx % 2
    if strategy == "switch_halfway":
        return 0 if step_idx < (total_steps // 2) else 1
    return 0 if label <= 4 else 1


async def _stream_demo_events(
    *,
    label: int,
    steps: int,
    seed: int,
    expert_urls: tuple[str, str],
    strategy: str,
) -> str:
    rng = np.random.default_rng(seed)
    x_t = rng.standard_normal((1, 32, 32, 1), dtype=np.float32)
    y = np.asarray([label], dtype=np.int32)
    clients = [ExpertClient(url, timeout_seconds=60.0) for url in expert_urls]

    start_payload = {
        "type": "start",
        "label": label,
        "steps": steps,
        "strategy": strategy,
        "frame": _normalize_frame(x_t),
        "selected_expert": None,
        "progress": 0.0,
    }
    yield f"data: {json.dumps(start_payload)}\n\n"

    dt = 1.0 / steps
    for step_idx in range(steps):
        selected_expert = _select_expert(step_idx, steps, label, strategy)
        t = np.full((1,), step_idx * dt, dtype=np.float32)
        velocity = clients[selected_expert].predict_velocity(x_t, t, y)
        x_t = x_t + dt * velocity
        payload = {
            "type": "step",
            "step": step_idx + 1,
            "steps": steps,
            "label": label,
            "selected_expert": selected_expert,
            "progress": float((step_idx + 1) / steps),
            "frame": _normalize_frame(x_t),
        }
        yield f"data: {json.dumps(payload)}\n\n"
        await asyncio.sleep(0.03)

    done_payload = {
        "type": "done",
        "label": label,
        "steps": steps,
        "selected_expert": _select_expert(steps - 1, steps, label, strategy),
        "progress": 1.0,
        "frame": _normalize_frame(x_t),
    }
    yield f"data: {json.dumps(done_payload)}\n\n"


def create_app() -> FastAPI:
    app = FastAPI(title="TPUGPU Router Demo")
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/demo/stream")
    async def stream_demo(
        label: int = 2,
        steps: int = 40,
        seed: int = 0,
        strategy: str = "alternating",
        expert_url_a: str = "http://34.162.118.249:8000",
        expert_url_b: str = "http://34.162.118.249:8000",
    ) -> StreamingResponse:
        return StreamingResponse(
            _stream_demo_events(
                label=label,
                steps=steps,
                seed=seed,
                expert_urls=(expert_url_a, expert_url_b),
                strategy=strategy,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return app
