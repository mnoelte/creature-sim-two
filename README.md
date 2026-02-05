# Creature Sim Two

Continuous 2D evolutionary simulation scaffold with headless and realtime modes. Creatures are built from parts (core, thruster, sensor, mouth, reproduction module). Fitness comes from gathering food and surviving; selection is tournament-based (k=4) with no elitism.

Project name: creature-sim-two

## Setup
1) Activate your virtual environment (Python 3.12+).
2) Install deps: `pip install -r requirements.txt`

## Usage
Headless (fast batches):
```
python main.py --mode headless
```

Realtime (pygame render + controls):
```
python main.py --mode realtime
```

Key controls (realtime):
- Space: pause/resume
- N: single-step one tick
- + / -: speed multiplier up/down
- M: switch mode (exits realtime loop; run headless next)

Notable parameters:
- `--tick-dt` (default 0.02): simulation step size (s)
- `--render-fps` (default 60): render cap; sim tick stays fixed
- `--population` (default 50)
- `--max-parts` (default 7)
- `--food-respawn-interval` (default 1.0s) and `--food-cap` (default 100)
- `--speed-multiplier` (default 1.0)
- `--seed` (default 42)
- `--generations` (default 10) and `--ticks-per-generation` (default 500)

## Notes
- Realtime requires `pygame`; headless runs without it.
- Current code is a scaffold: movement, collisions, sensing, reproduction, and stats collection are placeholders to be extended.
