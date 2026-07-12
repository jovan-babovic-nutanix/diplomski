from .maze import Maze, WALL, OPEN
from .generator import generate_maze
from .distance import bfs_distance_field, optimal_path_length
from .fixed_mazes import FIXED_MAZES, get_fixed_maze

__all__ = [
    "Maze",
    "WALL",
    "OPEN",
    "generate_maze",
    "bfs_distance_field",
    "optimal_path_length",
    "FIXED_MAZES",
    "get_fixed_maze",
]
