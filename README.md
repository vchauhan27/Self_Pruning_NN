# Self-Pruning Neural Network

A PyTorch implementation of a neural network that **learns to prune itself during training**  no post-training pruning step required. The network uses learnable gate parameters attached to every weight, and an L1 sparsity penalty that encourages most gates to collapse to exactly zero, effectively removing unnecessary connections on the fly.

Built for CIFAR-10 image classification.

---

## How It Works

Standard pruning removes weak weights *after* training. This project does it differently — every weight has a learnable **gate** (a scalar between 0 and 1) that multiplies the weight during the forward pass:

```
gates         = sigmoid(gate_scores)
pruned_weight = weight × gates
output        = input @ pruned_weight.T + bias
```

The network is trained with a combined loss:

```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```

Where `SparsityLoss` is the L1 norm (sum) of all gate values. This penalises the network for keeping gates open, forcing it to close gates on weights that don't justify their cost — pruning them automatically during training.


---

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy

## 🔧 Setup and Installation Guide

Instructions on how a user can get your project running on their own machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/vchauhan27/Self_Pruning_NN
    ```

2.  **Install `uv` (if you don't have it):**
    
    `uv` is a fast Python package manager used for this project. If you need to install it, run the appropriate command for your system.
    ```bash
    # On macOS, Linux, or Windows (WSL)
    curl -LsSf https://astral.sh/uv/install.sh| sh

    # On Windows (PowerShell)
    # You might have to first change your Execution Policy on your PC
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    
    #And then run this command to download uv
    irm https://astral.sh/uv/install.ps1| iex
    ```

3.  **Create and activate the virtual environment using `uv`:**
    * Make sure you are using Python version 3.12.
    * Download Python 3.12 if you dont have from
    * [Python 3.12](https://www.python.org/downloads/release/python-3120/)
    
    
    ```bash
    # Inside the Self_Pruning_NN2 folder initialize a uv project
    uv init
    # Create the virtual environment
    uv venv

    # Activate the environment
    # On Windows:
    .venv\Scripts\activate
    # On macOS/Linux:
    source .venv/bin/activate
    ```

4. **Install the required packages using `uv`:** 
    ```bash
    uv add -r requirements.txt
    ```
5. **Run the Code**
   ```bash
   python nn1.py
   ```

CIFAR-10 (~170MB) will be downloaded automatically into a `./data` folder on the first run.

---

## What the Script Does

1. Defines a custom `PrunableLinear` layer with learnable gate parameters
2. Builds a 4-layer feed-forward network using these layers
3. Trains the network on CIFAR-10 for three different values of λ
4. Reports test accuracy and sparsity level after each experiment
5. Generates two plots saved to the project folder

---

## Key Design Decisions

| Decision | Reason |
|---|---|
| Sigmoid for gates | Smooth and differentiable, maps any score to (0, 1) |
| L1 sparsity loss | Constant gradient pushes gates to exactly 0, not just small values |
| Gate scores init to +2.0 | `sigmoid(2) ≈ 0.88` — gates start open so sparsity has room to push them down |
| Gate LR boosted 50× | Gates receive small net gradients due to competing forces — boost ensures they move decisively |
| Raw sum (not mean) for sparsity loss | Keeps gradient magnitude high enough to overcome classification loss resistance |
| Small λ values (1e-06 to 5e-05) | Sparsity loss is a raw sum of ~1.7M gates ≈ 1,500,000 — λ must be tiny to balance against CrossEntropy ≈ 2.0 |

---

## Results

| Lambda (λ) | Test Accuracy | Sparsity Level |
|:----------:|:-------------:|:--------------:|
| 1e-06      | 54.87%        | 90.77%         |
| 8e-06      | 53.887%       | 98.58%         |
| 5e-05      | 49.20%        | 99.69%         |

The key finding: at **λ = 1e-06**, the network pruned **90.5% of all weights** while losing only ~1% accuracy. This shows the vast majority of connections in a flat network are redundant — the self-pruning mechanism successfully identifies and removes them.

---

## Output Plots

**gate_distribution.png** — Histogram of final gate values for the best model. A successful result shows a large spike at 0 (pruned weights) and a smaller cluster near 1 (surviving weights). The U-shape confirms the network is making decisive keep-or-kill decisions.

<img width="1138" height="632" alt="image" src="https://github.com/user-attachments/assets/003eb4ba-2e51-4ce3-b905-92cd67625638" />

---

**lambda_comparison.png** — Side-by-side bar chart comparing test accuracy and sparsity level across all three λ values, illustrating the sparsity-vs-accuracy trade-off.

<img width="1020" height="624" alt="image" src="https://github.com/user-attachments/assets/a982884c-cb7f-4214-be6b-9f93b9d33f15" />

---

## Limitations

- Uses a flat feed-forward network — accuracy is capped at ~55% on CIFAR-10 because spatial information in images is destroyed by flattening. A CNN-based version would achieve 80%+.
- Training on CPU is slow (~3-5 minutes per epoch). A CUDA-enabled GPU is strongly recommended.
- Results are sensitive to λ scale — since sparsity loss is a raw sum, λ values must be carefully chosen relative to network size.
