# Maze Navigation: GA vs NEAT vs Q-Learning vs A*

A bachelor's thesis project. A small grid agent (the "formula") navigates
procedurally generated mazes from a start cell to a goal cell. **Four methods**
are implemented and compared on identical environments:

* **A\*** (`src/planning/astar.py`) - classic informed search; always returns the
  shortest path. Used as the optimal **reference** the learning methods are
  measured against.
* **Genetic Algorithm (GA)** (`src/evolution/ga/`) - evolves a fixed-length
  *sequence of moves*. The agent replays the sequence; it has no perception.
  The easy-to-explain evolutionary baseline.
* **NEAT (NeuroEvolution of Augmenting Topologies)** (`src/evolution/neat/`) -
  implemented **from scratch**. Each agent has local *sensors* and a neural
  network "brain" whose weights *and topology* evolve (innovation tracking,
  crossover by historical markings, speciation with fitness sharing).
* **Q-Learning** (`src/rl/qlearning.py`) - tabular reinforcement learning. Learns
  a value `Q[state, action]` by trial and error with an epsilon-greedy policy and
  optional potential-based reward shaping.

All methods share the same maze, episode dynamics and evaluation, so the
comparison is fair. They are unified behind a common `Solver` interface
(`src/solver.py`).

## Comparison criteria (per the thesis)

1. **Success rate** - did the method reach the goal?
2. **Path length vs optimal** - found path length divided by the A* optimum.
3. **Convergence speed** - iterations and environment episodes until first solve.
4. **Execution time** - wall-clock time per run.

## Why the learning methods work: the fitness/reward gradient

"Did the agent reach the goal?" is a sparse signal. We precompute a **BFS
distance field** from the goal, turning progress into a smooth gradient (*how
many cells closer did the agent get?*). GA/NEAT use it in the shared fitness
(`src/fitness.py`); Q-Learning uses it for optional reward shaping.

## Project layout

```
config.py                 dataclass hyperparameters + seeding
src/
  maze/                   grid, seeded generator, fixed mazes, BFS distance
  environment/            agent, sensors, episode simulation
  fitness.py              shared fitness function
  solver.py               common Solver interface + EvolverSolver adapter
  planning/astar.py       A* (optimal reference) + AStarSolver
  rl/qlearning.py         tabular Q-Learning + QLearningSolver
  evolution/
    base.py               Evolver / Individual interfaces
    ga/                   genetic algorithm
    neat/                 NEAT from scratch
  visualization/          pygame renderer + interactive app
  experiments/            headless runner, metrics (CSV), matplotlib plots
scripts/
  run_game.py             interactive visualizer
  run_experiments.py      4-method comparison study
tests/                    pytest suite
```

## Setup

```bash
cd maze_evolution
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the interactive visualizer

```bash
python scripts/run_game.py --width 15 --height 15 --seed 42
```

Controls:

| Key       | Action                                          |
| --------- | ----------------------------------------------- |
| `g`       | Genetic Algorithm (population cloud + leader)   |
| `n`       | NEAT (population cloud + leader)                |
| `q`       | Q-Learning (single agent, greedy policy)        |
| `a`       | A* (single agent, optimal path)                 |
| `SPACE`   | pause / resume                                  |
| `+` / `-` | faster / slower animation                       |
| `r`       | new maze (new seed), restart all methods        |
| `v`       | toggle animation rendering (off = fast-forward) |
| `ESC`     | quit                                            |

For GA/NEAT, faded blue dots are the population, the gold marker is the current
best agent, and the gold line is the best path so far. For Q-Learning and A* a
single gold agent glides along the best/optimal path.

## Reproduce the thesis comparison

```bash
python scripts/run_experiments.py --seeds 5 --generations 80 --episodes 4000
```

Runs all four methods across several seeds (one freshly generated maze per seed,
shared by every method) and writes to `outputs/`:

* `history.csv` - per-iteration metrics (with cumulative environment episodes)
* `summary.csv` - per-trial outcomes (solved?, iterations/episodes-to-solve,
  wall-clock time, path length vs optimal)
* plots: `success_rate.png`, `optimality.png`, `convergence_iterations.png`,
  `convergence_evaluations.png`, `iterations_to_solve.png`, `wall_time.png`,
  `neat_complexity.png`

You can restrict methods, e.g. `--methods A*,GA,Q-Learning`. All randomness is
seeded, so results are reproducible.

## Tests

```bash
pytest -q
```

## Tunable parameters

All hyperparameters live in `config.py` (`MazeConfig`, `SimulationConfig`,
`FitnessConfig`, `GAConfig`, `NEATConfig`, `QLearningConfig`,
`ExperimentConfig`). Key ones are also exposed as command-line flags on the two
scripts.
