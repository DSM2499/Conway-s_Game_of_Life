# Conway-s_Game_of_Life

![Conway’s Game of Life](https://github.com/DSM2499/Conway-s_Game_of_Life/blob/main/problem-2/all_test.gif) ![Conway’s Game of Life](https://github.com/DSM2499/Conway-s_Game_of_Life/blob/main/problem-2/all_test.gif) 

**A high-performance simulation of Conway’s Game of Life extended with wormhole mechanics, validated by a self-supervised deep learning model.**  

This project showcases advanced systems design by integrating a rule-based, Numba-accelerated simulation engine with a custom-trained, unsupervised 3D convolutional network. The ML model serves as an automated validator that monitors the simulation’s physics—including the non-trivial wormhole teleportation behavior—enabling robust regression testing and rapid prototyping for complex cellular automata.

## Features

- **Rule-based Simulation Engine:**  
  Implements Conway’s Game of Life with custom wormhole behavior using Numba for parallel speed.

- **Wormhole Mechanics:**  
  Incorporates two types of wormholes (horizontal and vertical) defined through color-coded tunnel images, effectively “teleporting” cells between paired locations.

- **Self-supervised Validation:**  
  A custom-designed 3D convolutional network learns the “correct” evolution of the simulation in a self-supervised manner. It computes a per-frame binary cross-entropy (BCE) error to flag any deviations in the simulation logic.

- **Generic Checker and ML Generation:**  
  - **Checker:** Validates the simulation on the fly by comparing each generated frame against the model’s prediction.  
  - **ML Generator:** Optionally, an ML-based generator can autoregressively predict future frames, serving as a lightweight simulator.

- **Robust Data Handling:**  
  All input data is automatically padded to a common canvas, allowing the system to seamlessly handle multiple board sizes and ensuring consistency across training and inference.

- **High-Performance & Scalable:**  
  Utilizes efficient memory management (via streaming GIF readers and zero-padded datasets) and leverages CPU-optimized PyTorch for the ML components.

---

| Component | File | Highlights |
|-----------|------|------------|
| Rule engine | `game_of_life.py` | Numba‑parallel, supports redirect arrays for tunnels. |
| Tunnel mapping | `wormhole.py` | Reads two color PNGs and builds a 4‑D redirect array. |
| ML checker / simulator | `ML_checker.py` | 3‑channel Conv‑3D, automatic padding, logging‑only deviation detector. |
| Training script | `train_wormhole_checker.py` | Builds padded clips from trusted runs and trains `checker_wormhole.pt`. |
| Rule driver | `main.py` | Runs 1 000 steps, saves PNG snapshots & GIF, logs any physics drift. |
| ML generator (optional) | `ml_simulate.py` | Autoregressively generates frames with the Conv‑3D model. |

---

## Getting Started

### Prerequisites

- **Python ≥ 3.10**  
- **PyTorch (CPU-only version recommended)**  
- **Numba, imageio, OpenCV, Pillow, NumPy**

Install dependencies with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install numba imageio opencv-python pillow numpy
```
---

## Training the Wormhole-Aware Checker

The ML validator learns the correct simulation “physics” (including wormhole redirections) by training on a set of trusted simulation runs.

1. **Organize your data**
   - Place your tusted casese in the `data/` directory (each case must include `all.gif`).
  
2. **Run Training**
   ```bash
   cd src
   python/python3 train_wormhole_checker.py
   ```
   This script will:
   - Automatically build a padded dataset with 3 channels (life state, horizontal mask, vertical mask).
   - Train a 3D Convulational Neural Network (with kernel size matching the temporal window k = 4) using self-supervised learning.
   - Calibrate a deviation threshold (ε).
   - Save a checkpoint file checker_wormhole.pt containing the model’s weights, k, ε, and the target canvas size.

## Running the ML-Based Simulator

To generate a simulation entirely with the ML generator (autoregressive prediction):
```bash
python ml_simulate.py ../data/case-1
```
This mode uses the same learned wormhole-aware dynamics to predict future frames.

## Running a Simulation with Live Validation and Animation

To execute a full simulation on your test case (e.g., a case folder named problem-4) and generate snapshot images as well as an animated GIF summarizing the run, use the following command from the src/ directory:

```bash
python/python3 main.py ../problem-4 --animate
```
What this commad does:
- **Launches the Simulation**:
  
  The main.py script loads the input images (starting position, horizontal tunnel, and vertical tunnel) from the ../problem-4 directory and begins the simulation for a fixed number of steps (e.g., 1,000 steps).
- **Live Validation**:
  
  During simulation, each generated frame is checked on the fly using a trained ML-based validator. If any deviation from the expected simulation physics is detected (using a learned binary cross-entropy threshold), a warning is logged.
- **Snapshot Generation**:

    At specific simulation steps (e.g., steps 1, 10, 100, and 1000), the current state of the simulation is saved as PNG images.
- **Animate Output**:

    The `--animate` flag instructs the script to compile all the simulation frames into an animated GIF (`all_test.gif`), providing an overview of the entire simulation.

This command provides a comprehensive view of the simulation's performance and accuracy, making it easy to visually and quantitatively assess the system.

---

## Technical Insights

- **Data Preprocessing**:

    Each trusted simulation run is converted into a series of (k+1, 3, H, W) clips. Frames are zero-padded to the largest resolution (canvas) found across all cases. Two additional channels encode wormhole mappings derived from color-coded tunnel images.
- **Model Architecture**:

    A shallow 3D convolutional network (Conv3D) with input shape (B, 3, k, H, W) predicts the next life state frame with a sigmoid activation, trained with binary cross-entropy loss.
- **Self-Supervised Learning**:

    The model learns directly from simulation data with no manual labels. This enables automated anomaly detection: if the simulation drifts from expected behavior (measured via BCE error exceeding a threshold), a warning is logged.
- **Interoperability**:

    The ML checker integrates seamlessly with the rule-based engine to provide an extra safety net, ensuring that any changes to the simulation code are immediately flagged.
