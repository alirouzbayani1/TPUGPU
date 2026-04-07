#!/usr/bin/env python3
from __future__ import annotations

import argparse

import uvicorn

from tpugpu.demo import create_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TPUGPU router demo web app.")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
