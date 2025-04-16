from pathlib import Path
from ML_checker import fit_and_save

trusted_root = Path("data")          # contains case‑1, case‑2, …
cases = [p for p in trusted_root.iterdir() if (p / "all.gif").exists()]
fit_and_save(cases, k=4, out_file="checker_wormhole.pt", epochs=15)