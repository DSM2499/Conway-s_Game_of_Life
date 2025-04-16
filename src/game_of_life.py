import numpy as np
import time
from numba import njit, prange

@njit
def get_neighbors():
    """Returns relative offsets for the 8 neighboring cells."""
    return np.array([[-1, -1],
                     [-1,  0],
                     [-1,  1],
                     [ 0, -1],
                     [ 0,  1],
                     [ 1, -1],
                     [ 1,  0],
                     [ 1,  1]], dtype=np.int32)

@njit(parallel = True)
def simulate_step(grid, redirect_array, buffer):
    """
    Executes one step of Game of Life with wormhole support.

    Args:
        grid (np.ndarray): Current state of the grid.
        redirect_array (np.ndarray): Redirection map for wormholes.
        buffer (np.ndarray): Buffer for storing the next state.

    Returns:
        np.ndarray: Updated grid state.
    """
    h, w = grid.shape
    offsets = get_neighbors()
    max_iters = 1
    buffer[:, :] = 0

    for y in prange(h):
        for x in range(w):
            live_count = 0  # Initialize neighbor count for each cell.
            for i in range(8):
                 # Compute neighbor offset (ensure integer arithmetic)
                dy, dx = offsets[i]
                ny, nx = y + dy, x + dx

                new_y, new_x = ny, nx

                # Handle out-of-bounds explicitly using wormhole redirects
                if ny < 0:
                    tx = max(0, min(nx, w - 1))
                    if redirect_array[0, tx, 0, 0] != -1:
                        new_y, new_x = redirect_array[0, tx, 0]
                    else:
                        continue
                elif ny >= h:
                    tx = max(0, min(nx, w - 1))
                    if redirect_array[h - 1, tx, 2, 0] != -1:
                        new_y, new_x = redirect_array[h - 1, tx, 2]
                    else:
                        continue
                elif nx < 0:
                    ty = max(0, min(ny, h - 1))
                    if redirect_array[ty, 0, 3, 0] != -1:
                        new_y, new_x = redirect_array[ty, 0, 3]
                    else:
                        continue
                elif nx >= w:
                    ty = max(0, min(ny, h - 1))
                    if redirect_array[ty, w - 1, 1, 0] != -1:
                        new_y, new_x = redirect_array[ty, w - 1, 1]
                    else:
                        continue
                else:
                    # In-bounds cell → allow vertical + horizontal remapping
                    for _ in range(max_iters):
                        updated = False
                        if dy < 0 and redirect_array[new_y, new_x, 0, 0] != -1:
                            new_y, new_x = redirect_array[new_y, new_x, 0]
                            updated = True
                        elif dy > 0 and redirect_array[new_y, new_x, 2, 0] != -1:
                            new_y, new_x = redirect_array[new_y, new_x, 2]
                            updated = True
                        if not updated:
                            break

                    for _ in range(max_iters):
                        updated = False
                        if dx > 0 and redirect_array[new_y, new_x, 1, 0] != -1:
                            new_y, new_x = redirect_array[new_y, new_x, 1]
                            updated = True
                        elif dx < 0 and redirect_array[new_y, new_x, 3, 0] != -1:
                            new_y, new_x = redirect_array[new_y, new_x, 3]
                            updated = True
                        if not updated:
                            break

                # Count live neighbors
                if 0 <= new_y < h and 0 <= new_x < w:
                    live_count += grid[new_y, new_x]

            # Apply Game of Life rules
            if grid[y, x]:
                buffer[y, x] = 1 if live_count in (2, 3) else 0
            else:
                buffer[y, x] = 1 if live_count == 3 else 0
    
    return buffer