'''
Self-supervised checker for Game of Life with Wormholes.

Workflow:
1) Use 'fit_and_save()' to train a model from some "trusted" GIFs.
2) In your main simulation, create Checker("checker.pt") and call
   push_and_validate(frame) at each step.
3) If push_and_validate() returns (False, error_score), your
   simulation just produced a frame that's 'out-of-distribution'.

NOTE: If you have multiple board sizes, train one checker per size.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v2 as imageio
from collections import deque
from pathlib import Path
import logging

# ----------- Logging -----------------------------------------------------------------------
LOGGER = logging.getLogger("LifeChecker")
if not LOGGER.handlers:
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s  [%(levelname)s]  %(message)s",
        datefmt = "%H:%M:%S",
    )

# ----------- Padding to create a generic checker ------------------------------------------
def pad_to_canvas(frame: np.array, canvas_hw: tuple[int, int]) -> np.array:
    '''
    Zero‑pad 2‑D array 'frame' to (Hcanvas, Wcanvas).
    Assumes frame is smaller or equal in both dims.
    '''
    Hc, Wc = canvas_hw
    H, W = frame.shape
    return np.pad(frame,
                  ((0, Hc - H), (0, Wc - W)),
                  mode = "constant", constant_values = 0)

# ------------ Reading GIF's --------------------------------------------------------------
def gif_to_clips(path: Path, k: int, canvas_hw: tuple[int, int],
                 Hmask: np.ndarray, Vmask: np.ndarray):
    
    reader = imageio.get_reader(path, mode="I")
    buf = deque(maxlen = k + 1)

    for frame in reader:
        if frame.ndim == 3:
            frame = frame[..., 0]
        life = (frame > 127).astype(np.float32)
        life = pad_to_canvas(life, canvas_hw)
        buf.append(life)
        if len(buf) == k + 1:
            life_block = np.stack(buf)[:, None]
            masks = np.stack([Hmask, Vmask])
            masks = np.repeat(masks[None], k+1, 0)
            yield np.concatenate([life_block, masks], 1)

# ------------ Building masks --------------------------------------------------------------
def build_masks(h_map, v_map, Hc, Wc):
    Hmask = np.zeros((Hc, Wc), np.float32)
    Vmask = np.zeros((Hc, Wc), np.float32)

    for (y, x), (yy, xx) in h_map.items():
        if x < xx: Hmask[y, x], Hmask[yy, xx] = +1, -1
    for (y, x), (yy, xx) in v_map.items():
        if y < yy: Vmask[y, x], Vmask[yy, xx] = +1, -1
    return Hmask, Vmask

# ------------ Building dataset from multiple GIFs ------------------------------------------
def build_clip_dataset(cases: list[Path], k: int):
    max_h = max_w = 0

    for c in cases:
        f0 = imageio.get_reader(c / "all.gif", mode = "I").get_next_data()
        h0, w0 = f0.shape[:2]
        max_h, max_w = max(max_h, h0), max(max_w, w0)
    canvas = (max_h, max_w)

    # Iterate over every case
    from parser import load_color_image
    from wormhole import map_wormholes

    clips = []
    for case in cases:
        v_img = load_color_image(case / "horizontal_tunnel.png")
        h_img = load_color_image(case / "vertical_tunnel.png")
        H_mask, V_mask = build_masks(map_wormholes(h_img), map_wormholes(v_img), *canvas)
        for clip in gif_to_clips(case / "all.gif", k, canvas, H_mask, V_mask):
            clips.append(clip)
    return np.stack(clips, dtype = np.float32), canvas
            
# ------------ 3-D ConvPredictor ----------------------------------------------------------
class Conv3DPredictor(nn.Module):

    def __init__(self, k: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(3, 16, (k, 3, 3), padding=(0, 1, 1)),  #  ← 3 channels
            nn.ReLU(),
            nn.Conv3d(16, 32, 1), nn.ReLU(),
            nn.Conv3d(32,  1, 1), nn.Sigmoid(),
        )

    def forward(self, x): return self.net(x)[:, :, -1]

# ----------- Training functions ------------------------------------------------------------
def train_predictor(clips: np.ndarray, k: int, epochs = 15, batch_size = 64):
     # ------------- axis order fix -----------------------------------------
    X = torch.tensor(
            np.transpose(clips[:, :k], (0, 2, 1, 3, 4)),   # (N,3,k,H,W)
            dtype=torch.float32
        )
    Y = torch.tensor(
            clips[:, k, 0][:, None],                       # (N,1,H,W)
            dtype=torch.float32
        )
    # ----------------------------------------------------------------------

    ds = torch.utils.data.TensorDataset(X, Y)
    dl = torch.utils.data.DataLoader(ds, batch_size, shuffle=True)

    model = Conv3DPredictor(k)
    opt   = torch.optim.Adam(model.parameters(), 1e-3)
    bce   = nn.BCELoss()

    for ep in range(epochs):
        running = 0.0
        for xb, yb in dl:
            opt.zero_grad()
            loss = bce(model(xb), yb)
            loss.backward(); opt.step()
            running += loss.item() * xb.size(0)
        LOGGER.info(f"Epoch {ep:02d}  loss={running / len(dl.dataset):.4f}")

    return model

# ------------- Compute threshold by measuring average BCE ----------------------------------
def calibrate_eps(model, clips, k):
    X = torch.tensor(
            np.transpose(clips[:, :k], (0, 2, 1, 3, 4)),   # (N,3,k,H,W)
            dtype=torch.float32
        )
    Y = torch.tensor(
            clips[:, k, 0][:, None],                       # (N,1,H,W)
            dtype=torch.float32
        )

    bce, errs = nn.BCELoss(), []
    model.eval()
    with torch.no_grad():
        for xb, yb in zip(X, Y):
            errs.append(bce(model(xb[None]), yb[None]).item())

    errs = np.asarray(errs, dtype=np.float32)
    return float(errs.mean() + 4 * errs.std())

# ------------- fit and save -> Training pipeline from GIF's -> model + eps -----------------
def fit_and_save(cases: list[Path], k = 4, out_file = "checker_wormhole.pt", epochs = 15):
    clips, canvas = build_clip_dataset(cases, k)
    LOGGER.info(f"Canvas {canvas}, clips {clips.shape[0]}")
    model = train_predictor(clips, k, epochs)
    eps = calibrate_eps(model, clips, k)
    torch.save({
        "state_dict": model.state_dict(),
        "k": k,
        "eps": eps,
        "canvas": canvas
    }, out_file)
    LOGGER.info(f"Saved checker → {out_file}  (ε={eps:.4e})")

# ----------- Helper function to match prediction shape to target shape -------------------
def _match_hw(pred, tgt):
    ph, pw = pred.shape[-2:], tgt.shape[-2:]
    if ph[0] < tgt.shape[-2] or ph[1] < tgt.shape[-1]:
        pred = F.pad(pred, (0, tgt.shape[-1] - pw, 0, tgt.shape[-2] - ph))
    if pred.shape[-2] > tgt.shape[-2] or pred.shape[-1] > tgt.shape[-1]:
        dh = (pred.shape[-2] - tgt.shape[-2]) // 2
        dw = (pred.shape[-1] - tgt.shape[-1]) // 2
        pred = pred[..., dh: dh+tgt.shape[-2], dw: dw+tgt.shape[-1]]
    return pred

# ------------ Checker class ---------------------------------------------------------------
class Checker:
    def __init__(self, ckpt_path, name=None):
        ckpt = torch.load(ckpt_path, map_location = "cpu")
        self.k = ckpt["k"]       # how many frames to look back
        self.eps = ckpt["eps"]   # threshold
        self.canvas = ckpt.get("canvas")
        self.model = Conv3DPredictor(self.k)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        
        self.history = []
        self.scores = []
        self.flags = []

        self.name = name or Path(ckpt_path).stem

    def push_and_validate(self, frame_life: np.ndarray, Hmask, Vmask, step = -1):
        life = pad_to_canvas(frame_life, self.canvas)
        masks = np.stack([Hmask, Vmask])
        self.history.append(life)

        if len(self.history) <= self.k: 
            return True, 0.0
        
        inp = np.stack(self.history[-self.k:])
        mask_stack = np.repeat(masks[None], self.k, 0)

        x = np.transpose(
        np.concatenate([inp[:, None], mask_stack], 1),  # (k,3,H,W)
        (1, 0, 2, 3)                                    # → (3,k,H,W)
    )
        x = torch.tensor(x[None], dtype = torch.float32)

        with torch.no_grad():
            pred = self.model(x)
        
        tgt = torch.tensor(life[None, None], dtype = torch.float32)
        pred = _match_hw(pred, tgt)

        score = F.binary_cross_entropy(pred, tgt).item()
        self.scores.append(score)
        self.flags.append(score > self.eps)

        if score > self.eps:
            LOGGER.warning(f"[{self.name}] deviation at step {step}  BCE={score:.4e}")
        
        return score <= self.eps, score
            
    def summary(self):
        if not self.scores: return "No frames validated."
        arr, flg = np.array(self.scores), np.array(self.flags)
        return (f"Frames: {len(arr)} | Dev: {flg.sum()} ({100*flg.mean():.2f}%) | "
                f"Avg BCE: {arr.mean():.4e} | Max BCE: {arr.max():.4e}")
        