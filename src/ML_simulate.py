import os, logging, torch, numpy as np, imageio.v2 as imageio
from pathlib import Path
from ML_checker import Checker, Conv3DPredictor, pad_to_canvas, build_masks
from parser import load_binary_image, load_color_image
from wormhole import map_wormholes
from utils import save_binary_image, create_animation

logging.basicConfig(level = logging.INFO)
K, CKPT = 4, "checker_wormhole.pt"

def load_model():
    ckpt = torch.load(CKPT, map_location = "cpu")
    model = Conv3DPredictor(ckpt["k"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["canvas"]

def run_ml_sim(case_dir, steps = 1000, animate = True):
    case_dir = Path(case_dir)
    grid0 = load_binary_image(case_dir / "starting_position.png")
    h_img = load_color_image(case_dir / "horizontal_tunnel.png")
    v_img = load_color_image(case_dir / "vertical_tunnel.png")
    Hmask, Vmask = build_masks(map_wormholes(h_img), map_wormholes(v_img), *grid0.shape)

    model, canvas = load_model()
    history = [pad_to_canvas(grid0, canvas)]
    frames = [grid0.copy()]

    for step in range(1, steps):
        inp = np.stack(history[-K:])
        if inp.shape[0] < K: 
            inp = np.pad(inp, ((K-inp.shape[0], 0), (0, 0),(0, 0)))
        mask_stack = np.stack([Hmask, Vmask])
        mask_stack = np.repeat(mask_stack[None], K, 0)
        x = np.concatenate([inp[:, None], mask_stack], 1)
        
        with torch.no_grad():
            prob = model(torch.tensor(x[None], dtype = torch.float32))[0, 0].numpy()
        
        next_frame_pad = (prob>0.5).astype(np.float32)
        next_frame = next_frame_pad[:grid0.shape[0], :grid0.shape[1]]
        frames.append(next_frame); history.append(next_frame_pad)

        if step+1 in {1,10,100,1000}:
            save_binary_image(next_frame, case_dir / f"{step+1}.png")

        if animate: 
            create_animation(frames, case_dir/"all_test.gif", 50)
        logging.info("ML simulation done.")

if __name__ == "__main__":
    import sys
    if len(sys.argv)<2:
        print("Usage: python ml_simulate.py <case_dir>")
    else:
        run_ml_sim(sys.argv[1])