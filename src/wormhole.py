from collections import defaultdict
import numpy as np

def map_wormholes(image_array):
    """
    Maps wormhole positions from colored regions in the image.

    Args:
        image_array (np.ndarray): RGB image array.

    Returns:
        dict: Mapping of paired wormhole entry and exit coordinates.
    """
    color_map = defaultdict(list)
    h, w, _ = image_array.shape

    for y in range(h):
        for x in range(w):
            rgb = tuple(image_array[y, x])
            if rgb != (0, 0, 0):
                color_map[rgb].append((y, x))
    
    wormhole_pairs = {}

    for color, position in color_map.items():
        if len(position) == 2:
            a, b = position
            wormhole_pairs[a] = b
            wormhole_pairs[b] = a
    
    return wormhole_pairs

def build_redirect_array(h_map, v_map, height, width):
    """
     Constructs a redirection array indicating teleportation directions through wormholes.

    Args:
        h_map (dict): Horizontal wormhole map.
        v_map (dict): Vertical wormhole map.
        height (int): Height of the grid.
        width (int): Width of the grid.

    Returns:
        np.ndarray: 4D array encoding redirection mappings per cell and direction.
    """
    redirect_array = np.full((height, width, 4, 2), -1, dtype = np.int32)

    def _set_redirect(y, x, dir_idx, ny, nx):
        if 0 <= y < height and 0 <= x < width:
            redirect_array[y, x, dir_idx, 0] = ny
            redirect_array[y, x, dir_idx, 1] = nx

    for a, b in h_map.items():
        ay, ax = a
        by, bx = b
        
        #a to b
        _set_redirect(ay, ax, 1, by, (bx + 1) % width) #%width
        _set_redirect(ay, ax, 3, by, (bx - 1) % width) #%width

        #b to a
        _set_redirect(by, bx, 1, ay, (ax + 1) % width) #%width
        _set_redirect(by, bx, 3, ay, (ax - 1) % width) #%width
    
    for a, b in v_map.items():
        ay, ax = a
        by, bx = b

        #a to b
        _set_redirect(ay, ax, 0, (by - 1) % height, bx) #(by - 1) % height
        _set_redirect(ay, ax, 2, (by + 1) % height, bx) #(by + 1) % height

        #b to a
        _set_redirect(by, bx, 0, (ay - 1) % height, ax) #(ay - 1) % height
        _set_redirect(by, bx, 2, (ay + 1) % height, ax) #(ay + 1) % height

    return redirect_array


        
        