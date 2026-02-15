from __future__ import annotations

import argparse

from config import Mode, SimulationConfig


def parse_args() -> SimulationConfig:
    parser = argparse.ArgumentParser(
        description="2D evolving creatures simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=[m.value for m in Mode],
        default=Mode.REALTIME.value,
        help="Execution mode: headless (no window) or realtime (pygame window)",
    )
    parser.add_argument("--tick-dt", type=float, default=0.02, help="Simulation tick size in seconds")
    parser.add_argument("--render-fps", type=int, default=60, help="Max render frames per second (realtime mode)")
    parser.add_argument("--population", type=int, default=50, help="Population size per generation")
    parser.add_argument("--max-parts", type=int, default=7, help="Maximum parts per creature genome")
    parser.add_argument(
        "--food-respawn-interval",
        type=float,
        default=1.0,
        help="Seconds between food respawn checks",
    )
    parser.add_argument("--food-cap", type=int, default=100, help="Maximum food items present")
    parser.add_argument("--speed-multiplier", type=float, default=1.0, help="Global sim speed multiplier")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for determinism")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations to run")
    parser.add_argument(
        "--ticks-per-generation",
        type=int,
        default=500,
        help="Simulation ticks per generation",
    )
    args = parser.parse_args()

    return SimulationConfig(
        mode=Mode(args.mode),
        tick_dt=args.tick_dt,
        render_fps=args.render_fps,
        population_size=args.population,
        max_parts=args.max_parts,
        food_respawn_interval=args.food_respawn_interval,
        food_cap=args.food_cap,
        speed_multiplier=args.speed_multiplier,
        seed=args.seed,
        generations=args.generations,
        ticks_per_generation=args.ticks_per_generation,
    )
