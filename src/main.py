import os, logging, numpy as np
from parser    import load_binary_image, load_color_image
from wormhole  import map_wormholes, build_redirect_array
from game_of_life import simulate_step
from utils     import save_binary_image, create_animation
from ML_checker import Checker, build_masks
from ML_checker import pad_to_canvas

logging.basicConfig(level=logging.INFO)

def run_sim(input_dir, animate=False):
    g = load_binary_image(os.path.join(input_dir,"starting_position.png"))
    h = load_color_image(os.path.join(input_dir,"horizontal_tunnel.png"))
    v = load_color_image(os.path.join(input_dir,"vertical_tunnel.png"))
    redirect = build_redirect_array(map_wormholes(h), map_wormholes(v), *g.shape)
    Hmask, Vmask = build_masks(map_wormholes(h), map_wormholes(v), *g.shape)
    checker = Checker("checker_wormhole.pt")
    
    Hmask_pad = pad_to_canvas(Hmask, checker.canvas)
    Vmask_pad = pad_to_canvas(Vmask, checker.canvas)

  
    state, buffer = g.copy(), np.zeros_like(g)
    frames=[]; png_steps={1,10,100,1000}

    for i in range(1,1001):
        simulate_step(state, redirect, buffer); state, buffer = buffer, state
        checker.push_and_validate(state, Hmask_pad, Vmask_pad, step=i)
        frames.append(state.copy())
        if i in png_steps: save_binary_image(state, f"{input_dir}/{i}.png")

    if animate: create_animation(frames, f"{input_dir}/all_test.gif",50)
    print("\n=== Accuracy summary ==="); print(checker.summary())

if __name__=="__main__":
    import sys
    if len(sys.argv)<2: print("python main.py <case_dir> [--animate]")
    else: run_sim(sys.argv[1], "--animate" in sys.argv[2:])