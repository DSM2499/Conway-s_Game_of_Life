# train_generic_checker.py
from pathlib import Path
from ML_checker import fit_and_save

# point to every directory of trusted GIFs you already have
gif_dirs = [
    Path("data/good_runs_45x62"),
    Path("data/good_runs_409x436"),
    # add more as you collect them
]

gif_paths = [p for d in gif_dirs for p in d.glob("*.gif")]
fit_and_save(
    gif_paths=sorted(gif_paths),
    k=4,
    out_file="checker_generic.pt",
    epochs=15,
)