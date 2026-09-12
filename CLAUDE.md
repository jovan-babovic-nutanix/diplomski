# Project context: Maze Navigation thesis (GA vs Q-Learning vs A*)

This is Jovan Babovic's bachelor's thesis project at ETF Belgrade (advisor: Vladimir).
Read this file fully before making changes — it captures decisions and history that
aren't visible from the code alone.

## What the professor actually assigned

Topic: "Poredjenje genetskog algoritma i metoda ucenja sa podrskom (RL) u problemu
navigacije agenta kroz lavirint" — compare a Genetic Algorithm, Q-Learning
(reinforcement learning), and A* (as the optimal reference baseline) on maze
navigation, judged on: success rate, path length, convergence speed, execution time.

**NEAT/neuroevolution was NOT requested by the professor.** It was implemented early
on as a bonus 4th method, caused real problems (got stuck near the start as a
memory-less reactive policy, ~0% success on larger mazes, very slow), and was
**deliberately removed entirely** later once we confirmed the professor didn't ask
for it. Do not re-add NEAT unless Jovan explicitly asks for it again — there is no
`src/evolution/neat/` anymore and no reason to bring it back for the thesis.

The final method set is exactly three: **A\*** (reference/optimal), **GA**, **Q-Learning**.

## Architecture (matches README.md, which is accurate for this)

- `config.py` — all hyperparameters as dataclasses (`MazeConfig`, `SimulationConfig`,
  `FitnessConfig`, `GAConfig`, `QLearningConfig`, `ExperimentConfig`) + seeding.
- `src/maze/` — grid maze, seeded recursive-backtracker generator, 3 fixed mazes
  (`empty`, `simple`, `spiral`), BFS distance field from the goal (`distance.py`).
- `src/environment/` — grid agent + episode simulator shared by all methods.
- `src/fitness.py` — shared GA fitness: `distance_weight * progress + goal_bonus (if
  reached) - step_penalty * steps (if reached) - bump_penalty * bumps`. Gives partial
  credit for getting closer, not just solved/unsolved, so GA has a usable gradient.
- `src/planning/astar.py` — A*, Manhattan heuristic, optimal reference.
- `src/rl/qlearning.py` — tabular Q-Learning, epsilon-greedy, optional BFS-based
  potential reward shaping.
- `src/evolution/` — GA only (`ga/genome.py`, `ga/ga.py`, `base.py` Evolver interface).
  **GA genome is an open-loop fixed-length sequence of moves** (length = max_steps).
  The agent does NOT perceive the maze while replaying it — this is intentional and
  is a known, discussed limitation (large mazes are hard for GA because it's
  searching a huge space of full action sequences, not reacting to walls).
- `src/solver.py` — common `Solver` interface (`step()`, `best_path()`,
  `is_converged()`, plus optional `population_trajectories()`/
  `leader_trajectory()` hooks used only by GA) so GA/Q-Learning/A* can be driven
  uniformly by the runner and UI. One `step()` = one GA generation / one
  Q-Learning training batch / one A* planning step.
- `src/experiments/` — headless `runner.py` (fair comparison: same maze per seed for
  all 3 methods), `metrics.py` (writes `history.csv`, `summary.csv`), `plots.py`
  (success_rate, optimality, convergence x2, iterations_to_solve, wall_time).
- `src/visualization/` — pygame `app.py` + `renderer.py`, plus `session.py`
  (`ComparisonSession`): the round-advancing/completion/metrics state machine,
  with no pygame import, so it's testable without a display. `MazeApp` is a
  thin pygame wrapper (init/input/rendering/frame timing) that delegates to a
  `ComparisonSession`. GA is driven through `EvolverSolver` exactly like
  Q-Learning/A*, not reimplemented separately — see "2026 audit & fixes" below.
- `scripts/run_game.py` (interactive) and `scripts/run_experiments.py` (headless
  study) are the two entry points. Both accept `--preset demo|thesis` (see
  `config.py`'s `demo_config()`/`thesis_config()`), with every individual flag
  still able to override a value from the chosen preset.
- `tests/` — 37 tests, `pytest -q`. Keep this passing after any change.

## Action encoding — do not change casually

```python
MOVES = ((-1, 0), (0, 1), (1, 0), (0, -1))  # 0=Up, 1=Right, 2=Down, 3=Left (clockwise)
MOVE_NAMES = ("U", "R", "D", "L")
```
This was deliberately changed to clockwise order at Jovan's request and is now used
consistently by GA, Q-Learning, A*, and the simulation/agent — verified with an
explicit test. If you ever touch `MOVES`, re-audit every consumer the same way.

## Behavioral decisions worth knowing (so you don't "fix" intentional behavior)

- **Fixed computational budget, not "run until solved."** For fair experiments, GA
  and Q-Learning must have a hard cap (generations / episodes) and are recorded as
  failed if they don't solve within it — otherwise success-rate comparisons are
  meaningless. Current GA config: `population_size=300, generations=200` (raised
  from 200/80 after testing). The *interactive UI* also stops each method at its cap
  or on success — it must never run forever.
- **GA finishes the whole generation before stopping**, even if one agent already
  solved it mid-generation — needed for correct generation statistics (best/mean
  fitness, success rate for that generation). This is intentional, not a bug.
- **Q-Learning bug (fixed):** it used to keep training after its greedy rollout had
  already found the goal, so the UI path kept changing. Fixed so `step()` stops
  training and `is_converged()` returns True once solved; the UI freezes the
  successful path. There's a regression test for this — don't regress it.
- **Fast-forward (`v` key) needed a hard cap too** — Q-Learning fast-forwarding used
  to be able to spin forever if it never solved. It's now capped at
  `episodes / episodes_per_step` rounds, and the whole app has a `FINISHED` state
  once GA and Q-Learning both stop (solved or budget exhausted); `r` resets with a
  new maze.
- **Visual speed cap:** at high speed multipliers (UI supports up to ~300x) agent
  dots must not visually jump straight to the end of their trajectory — movement is
  capped at ~8 cells/frame so agents stay visible even sped up. `v` toggles between
  animated and true fast-forward (no per-frame drawing).
- **UI visual style** went through several iterations at Jovan's request: originally
  a dark/checkerboard theme -> a "Flying Dots" redesign matching the reference repo
  github.com/pantela002/GENETIC-ALGORITHM-MAZE exactly -> briefly a 2x2 parallel
  panel layout while NEAT still existed -> a horizontal 3-panel layout (GA /
  Q-Learning / A*) since NEAT's removal -> a 2026 WCAG-contrast/z-order pass on the
  Flying Dots palette -> **now (2026) a full "Race Control" redesign**: a dark
  motorsport/mission-control HUD look, chosen from three mocked-up directions (see
  `scratch/maze-arena-directions.html`, `data-dir="a"` — the other two, "Arcade
  Tournament" and "Research Console", were not chosen but are kept there for
  reference). All color/font/shape constants live at the top of
  `src/visualization/renderer.py`; real font files (Chakra Petch, IBM Plex Sans, IBM
  Plex Mono — all SIL OFL) are bundled under `assets/fonts/` with a graceful
  `pygame.font.SysFont` fallback (see `load_font()`) if that directory is ever
  missing, so tests/CI never depend on the asset files being present. Every color
  constant is contrast-checked in `tests/test_visual_contrast.py` — if you change a
  color there and a test fails, fix the color, don't lower the threshold. Panels are
  cut-corner cards with a "LEARNING"/"REFERENCE" tag badge (static per method: GA
  and Q-Learning are LEARNING, A* is REFERENCE — this is not derived from solver
  state) and a colored status pill (running/solved/failed) instead of the old
  single-line stat text. Glow effects (leader marker, best-path line, goal marker,
  panel head dot) are hand-rolled via concentric alpha circles (`_draw_glow`) since
  pygame has no blur primitive — deliberately **not** applied to the GA population
  swarm itself (300 individuals at 60fps; verified draw-only throughput stays
  >400fps without swarm glow, and glowing every dot would wash the swarm into a
  blur anyway). The maze floor/wall fill colors are intentionally low-contrast with
  each other (keeps the HUD dark enough for the glow to read) — maze structure is
  instead carried by a `WALL_EDGE` boundary outline drawn only where wall meets
  floor; don't "fix" this by brightening the wall fill without re-reading why.
- **`NEXT ROUND` button + `N` key** advance exactly one round for every active method
  at once (useful for stepping through slowly during a demo/defense); the button
  now visibly grays out once everything is `FINISHED`. The whole control bar is
  clickable (`Renderer.control_rects()` + `MazeApp._handle_click`), not just that
  one button: Pause, Next Round, speed +/-, New Maze, Fast-Forward, and a
  **"Show population"** checkbox (`P` key or click) that hides GA's swarm dots for a
  cleaner screenshot — the leader marker stays visible either way, so the panel is
  never empty.
- **`LEADERBOARD` side panel** (renamed from `FINAL RESULTS`) shows three bar-chart
  groups (path/optimal, time to solve, evaluations) plus a per-method detail line.
  It still only fills in a method's row once that method's metrics are genuinely
  finalized (`ComparisonSession._record_completion` / `metrics[m]["completed"]`) —
  this was a deliberate earlier design decision (see the note in
  `Renderer.draw_summary`'s docstring) and the 2026 redesign kept it: never a
  misleading "live" partial comparison, just real numbers filled in as each method
  actually finishes. `ComparisonSession._record_completion` also now stores a raw
  `time_s_val` float alongside the formatted `time_s` string, purely so the sidebar
  can compute bar widths without re-parsing a formatted string — this is the one
  session.py change the redesign needed; no round-advancing/completion logic
  changed.

## Environment note

Jovan previously developed this on a Mac; he is now on Windows. Setup there:

```powershell
cd C:\Users\iohl\Desktop\diplomski
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\run_game.py --width 15 --height 15 --seed 42
pytest -q
python scripts\run_experiments.py --seeds 5 --generations 80 --episodes 4000
```

The README's "cd maze_evolution" instruction is stale — the project root already
contains `src/`, `scripts/`, `config.py` etc. directly; there is no `maze_evolution/`
subdirectory.

## 2026 audit & fixes (read this if touching metrics, config, or the UI)

A UI/correctness audit found and fixed several bugs; the details matter if you
touch these areas again.

- **`StepStats.iteration` is 0-indexed for every method, including
  Q-Learning.** It used to report 1 on Q-Learning's first `step()` (an
  off-by-one against GA's convention and against `run_solver`'s own loop
  index), which made `summary.csv`'s `iterations_to_solve` disagree with the
  matching row of `history.csv`. Fixed via an explicit `self._round` counter
  in `QLearningSolver`, independent of the episode-count arithmetic. If you
  ever touch iteration/generation numbering again, re-verify
  `tests/test_experiment.py::test_iterations_to_solve_matches_history_row`.
- **`StepStats.evaluations` accounting convention** (see `src/solver.py`'s
  docstring): every full environment episode counts once. Q-Learning's
  `evaluations` now includes the per-round greedy-rollout episode, not just
  training episodes — it used to undercount by one episode per round.
- **`q_coverage`** (Q-Learning's extra stat) is a 0–1 fraction of the maze's
  open cells with a learned Q-value, not a raw cell count.
- **A*'s "done" state must come from `is_converged()`, not from
  reachability.** The interactive UI used to derive `astar_done` from whether
  a path was found, which would have hung the app forever on an unreachable
  goal (not triggerable today since the maze generator guarantees
  connectivity, but a real trap for hand-authored fixed mazes).
- **GA now goes through `EvolverSolver`** (`src/solver.py`) in the interactive
  UI too, via the population-trajectory hooks on `Solver`/`Evolver`, instead
  of `MazeApp` reimplementing generation-stepping separately. Two small,
  accepted behavior changes came from this: a GA panel's "solved" badge is now
  sticky across the whole run (was per-generation), and its displayed "steps"
  is the best-ever path length (was the current generation's best) — both
  already matched how Q-Learning/A* and the headless runner behaved.
- **Renderer palette values were tuned for WCAG contrast** (`src/visualization/
  renderer.py`): several "Flying Dots" reference colors (start marker, leader
  ring, best-path line, some text/accent colors) were too low-contrast against
  the cyan maze floor or panel chrome. Current values are verified with
  `tests/test_visual_contrast.py` — don't lower a color's contrast below what
  that test checks without updating it deliberately.
- **`demo_config()` / `thesis_config()` presets** in `config.py` formalize what
  used to be an undocumented divergence between `run_game.py`'s defaults
  (small/fast) and `config.py`'s dataclass defaults (large/slow, used by
  `run_experiments.py`). `thesis_config()` uses `n_seeds=20` (was 5 by
  default) for a more defensible success-rate comparison.
- **Open methodology questions, not yet decided** (no code needed, just a
  decision before the final thesis numbers are frozen): whether to run GA and
  Q-Learning at an equal episode budget as a fairness check (currently GA gets
  roughly 15x more environment episodes than Q-Learning at default settings);
  whether the headline Q-Learning numbers should use `reward_shaping=True` or
  `False` (shaping uses the same BFS distance field as GA's fitness, just
  injected more densely); and when to run the maze-size scaling study.

## Ideas discussed but not yet implemented (possible next steps)

- A maze-size scaling study (e.g. 11x11 / 21x21 / 31x31) to show how each method's
  success rate/time/path-length scales — flagged as a strong thesis addition.
- Equalizing the evaluation "budget" (total environment episodes) more precisely
  across GA and Q-Learning for the fairest possible comparison.
- A Q-Learning value/policy heatmap overlay in the visualizer — flagged as effective
  for the thesis defense but not built.
- Sensitivity analysis on key hyperparameters (Q-Learning alpha/gamma, GA
  mutation_rate) for one thesis section.

## Repo

Public GitHub repo: https://github.com/jovan-babovic-nutanix/diplomski
