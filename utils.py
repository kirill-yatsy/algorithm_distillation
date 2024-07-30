 
import numpy as np

def prepare_rgb_image_to_print_in_terminal(grid):
    grid = grid.astype(int)
    grid = np.where(grid == 255, " ", grid)
    grid = np.where(grid == '1', "X", grid)
    grid = np.where(grid == '2', "G", grid)
    return grid

def print_grid(grid):
    print("\033[H\033[J")
    grid = prepare_rgb_image_to_print_in_terminal(grid)
    print(grid)