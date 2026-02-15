"""Entry point for the 2D evolutionary simulation scaffold."""

from __future__ import annotations

from cli import parse_args
from config import Mode
from simulator import Simulator


def main() -> None:
    config = parse_args()
    simulator = Simulator(config)

    if config.mode == Mode.HEADLESS:
        simulator.run_headless()
    else:
        simulator.run_realtime()


if __name__ == "__main__":
    main()
