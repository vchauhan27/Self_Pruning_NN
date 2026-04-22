import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

## Defining the PrunableLinear layer
class PrunableLinear(nn.Module):
    """
    A custom layer that prunes weights according to their relevance.
    Forward Propagation:
        gates = sigmoid(gate_scores)
        pruned_weight = weight * gates
        output = input * pruned_weight + bias
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.gate_scores = nn.Parameter(torch.empty(out_features, in_features))

        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)

        # Initialize gate_scores to 2.0 so sigmoid(2.0) ≈ 0.88.
        nn.init.constant_(self.gate_scores, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores).detach()

## Defining the actual Neural Network
class SelfPruningNet(nn.Module):
    """
    A 3-layer feed-forward network for CIFAR-10.
    The input in this dataset is in shape 32x32x3 = 3072 input features.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = PrunableLinear(3072, 512)
        self.fc2 = PrunableLinear(512, 256)
        self.fc3 = PrunableLinear(256, 128)
        self.fc4 = PrunableLinear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)  # Flattening (B, 32,32,3) -> (B,3072)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def get_all_gates(self) -> torch.Tensor:
        """Calculate all the gate values across every hidden layer."""
        gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                gates.append(module.gates().flatten())
        return torch.cat(gates)

def sparsity_loss(model):
    """
    L1 Normalization of all gate values.
    FIX: We return the SUM, not the mean. Dividing by numel() scales the
    gradient down so much that the optimizer ignores it.
    """
    gates_list = []
    for module in model.modules():
        if isinstance(module, PrunableLinear):
            gates = torch.sigmoid(module.gate_scores)
            gates_list.append(gates.flatten())

    all_gates = torch.cat(gates_list)
    return all_gates.sum()  


def get_cifar10_loaders(batch_size: int = 128):
    """ Return train and test dataloaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616)),
    ])

    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader

## Training Loop
def train_one_epoch(model, loader, optimizer, lambda_sparse, device):
    """Forward Prop on the neural net we have made."""
    model.train()
    total_cls_loss = 0
    total_spar_loss = 0
    total_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)

        cls_loss = F.cross_entropy(logits, labels)
        spar_loss = sparsity_loss(model)

        loss = cls_loss + lambda_sparse * spar_loss

        loss.backward()
        optimizer.step()

        total_cls_loss += cls_loss.item()
        total_spar_loss += spar_loss.item()
        total_loss += loss.item()

    n = len(loader)
    return total_loss / n, total_cls_loss / n, total_spar_loss / n

# Evaluation
def evaluate(model, loader, device):
    """Calculating model accuracy"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total

def compute_sparsity(model, threshold: float = 1e-2) -> float:
    """Gate < threshold is practically pruned"""
    all_gates = model.get_all_gates()
    pruned = (all_gates < threshold).sum().item()
    return 100 * pruned / all_gates.numel()

## Finally running the model for one lambda
def run_model(lambda_sparse: float,
              train_loader,
              test_loader,
              device,
              epochs: int = 15,
              lr: float = 1e-3) -> dict:

    """Training the model"""
    print(f"\nTraining with Lambda = {lambda_sparse}")

    model = SelfPruningNet().to(device)

    #Using different parameters for gate and weights
    gate_params = []
    standard_params = []

    for name, param in model.named_parameters():
        if "gate_scores" in name:
            gate_params.append(param)
        else:
            standard_params.append(param)

    # Give standard weights the normal lr, but give gates a 50x boost
    optimizer = optim.Adam([
        {'params': standard_params, 'lr': lr},
        {'params': gate_params, 'lr': lr * 50}
    ])
    # ---------------------------------------------------------

    for epoch in range(1, epochs + 1):
        loss, cls, spar = train_one_epoch(model, train_loader, optimizer, lambda_sparse, device)

        if epoch % 5 == 0 or epoch == 1:
            acc = evaluate(model, test_loader, device)
            sparsity = compute_sparsity(model)
            print(f"  Epoch {epoch:>2}/{epochs}  |  "
                  f"Loss: {loss:.4f}  |  "
                  f"Acc: {acc:.2f}%  |  "
                  f"Sparsity: {sparsity:.2f}%")

    final_acc = evaluate(model, test_loader, device)
    final_sparsity = compute_sparsity(model)
    final_gates = model.get_all_gates().cpu().numpy()

    print(f"\n  Final Test Accuracy : {final_acc:.2f}%")
    print(f"  Final Sparsity Level: {final_sparsity:.2f}%")

    return {
        "lambda":   lambda_sparse,
        "accuracy": final_acc,
        "sparsity": final_sparsity,
        "gates":    final_gates,
        "model":    model
    }

## Plotting
def plot_gate_distribution(results: list, best_idx: int):
    """Plot the gate value distribution for the best model."""
    best = results[best_idx]
    gates = best["gates"]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_yscale('log')
    ax.hist(gates, bins=80, color="#64E0B1", edgecolor="white", linewidth=0.3)
    ax.set_title(
        f"Gate Value Distribution  |  λ = {best['lambda']}  "
        f"|  Sparsity = {best['sparsity']:.1f}%  "
        f"|  Accuracy = {best['accuracy']:.1f}%",
        fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Gate Value  (0 = pruned,  1 = fully active)", fontsize=11)
    ax.set_ylabel("Number of Weights", fontsize=11)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.2,
               label="Prune threshold (0.01)")
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("gate_distribution.png", dpi=150)
    plt.show()
    print("\n  [Plot saved as gate_distribution.png]")

def plot_results_summary(results: list):
    """Bar chart comparing accuracy and sparsity across λ values."""
    lambdas   = [str(r["lambda"]) for r in results]
    accs      = [r["accuracy"]    for r in results]
    sparsities= [r["sparsity"]    for r in results]

    x = np.arange(len(lambdas))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - width/2, accs,       width, label="Test Accuracy (%)",  color="#2563EB", alpha=0.85)
    bars2 = ax2.bar(x + width/2, sparsities, width, label="Sparsity Level (%)", color="#16A34A", alpha=0.85)

    ax1.set_xlabel("Lambda (λ)", fontsize=12)
    ax1.set_ylabel("Test Accuracy (%)", color="#DB5555", fontsize=11)
    ax2.set_ylabel("Sparsity Level (%)", color="#C452E7", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"λ={l}" for l in lambdas])
    ax1.set_ylim(0, 100)
    ax2.set_ylim(0, 100)

    fig.legend(loc="upper right", bbox_to_anchor=(0.88, 0.88), fontsize=10)
    plt.title("Sparsity vs Accuracy Trade-off Across λ Values", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("lambda_comparison.png", dpi=150)
    plt.show()
    print("  [Plot saved as lambda_comparison.png]")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_cifar10_loaders(batch_size=128)

    lambdas = [1e-6, 8e-6, 5e-5]

    results = []
    for lam in lambdas:
        result = run_model(lam, train_loader, test_loader, device, epochs=15)
        results.append(result)

    print("\n")
    print("=" * 52)
    print(f"  {'Lambda':<12} {'Test Accuracy':>15} {'Sparsity (%)':>15}")
    print("-" * 52)
    for r in results:
        print(f"  {r['lambda']:<12} {r['accuracy']:>14.2f}% {r['sparsity']:>14.2f}%")
    print("=" * 52)

    best_idx = max(range(len(results)), key=lambda i: results[i]["accuracy"])
    plot_gate_distribution(results, best_idx)
    plot_results_summary(results)

if __name__ == "__main__":
    main()