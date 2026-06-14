"""
Part 1: Train a Custom Network on CIFAR-10 (60%)

Experiments:
  E1: Different filter counts  (S/M/L)
  E2: Different activations     (ReLU / LeakyReLU)
  E3: Different loss functions  (with/without L2 regularization)
  E4: Different optimizers      (Adam / SGD+momentum / RMSprop)

Output: ../reports/figures/part1_*.png and ../reports/models/part1_*.pth
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
import random
from tqdm import tqdm

from models.my_cifar_net import MyCIFARNet, MyCIFARNet_S, MyCIFARNet_M, MyCIFARNet_L
from data.loaders import get_cifar_loader

# ========== Config ==========
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
SEED = 2020

module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== Seed ==========
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ========== Data ==========
print("Loading CIFAR-10...")
train_loader = get_cifar_loader(train=True, batch_size=BATCH_SIZE)
val_loader = get_cifar_loader(train=False, batch_size=BATCH_SIZE)
test_loader = get_cifar_loader(train=False, batch_size=BATCH_SIZE)  # CIFAR-10 test = val
print(f"Train: {len(train_loader)} batches, Test: {len(test_loader)} batches")


# ========== Helpers ==========
def get_accuracy(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def train_one_model(model, optimizer, criterion, train_loader, val_loader, epochs, model_name):
    """Train a single model and return curves."""
    model.to(device)
    train_losses = []
    train_accs = []
    val_accs = []
    best_val_acc = 0
    best_epoch = 0

    pbar = tqdm(range(epochs), unit='epoch', desc=model_name)
    for epoch in pbar:
        # --- Train ---
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # --- Eval ---
        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(),
                       os.path.join(models_path, f'{model_name}_best.pth'))

        pbar.set_postfix({
            'loss': f'{avg_loss:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })

    # Load best and get test accuracy
    model.load_state_dict(torch.load(os.path.join(models_path, f'{model_name}_best.pth')))
    test_acc = get_accuracy(model, test_loader)
    print(f"  {model_name}: Best val acc = {best_val_acc:.4f} (epoch {best_epoch+1}), Test acc = {test_acc:.4f}")

    return train_losses, train_accs, val_accs, best_val_acc, test_acc


SKIP_E1_E3 = False  # True=跳过E1/E2/E3, 只跑E4优化器对比

# ================================================================
#  E1: Different Filter Counts (S=16, M=32, L=64)
# ================================================================
if not SKIP_E1_E3:
    print("\n" + "=" * 50)
    print("Experiment E1: Different Filter Counts (S=16, M=32, L=64)")
    print("=" * 50)

    models_e1 = {
        'S (base_ch=32, ~2.9M)': (MyCIFARNet_S, {'activation': 'relu'}),
        'M (base_ch=48, ~6.5M)': (MyCIFARNet_M, {'activation': 'relu'}),
        'L (base_ch=64, ~11M)': (MyCIFARNet_L, {'activation': 'relu'}),
    }

    results_e1 = {}
    for name, (net_fn, kwargs) in models_e1.items():
        set_seed(SEED)
        model = net_fn(**kwargs)
        params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"\nTraining {name} [{params:.2f}M params]")
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        curves = train_one_model(model, opt, crit, train_loader, val_loader, EPOCHS, f'part1_E1_{name}')
        results_e1[name] = curves
else:
    results_e1 = {}


# ================================================================
#  E2: Different Activations (ReLU vs LeakyReLU)
# ================================================================
if not SKIP_E1_E3:
    print("\n" + "=" * 50)
    print("Experiment E2: Different Activations (ReLU vs LeakyReLU)")
    print("=" * 50)

    results_e2 = {}
    for act in ['relu', 'leaky_relu']:
        set_seed(SEED)
        model = MyCIFARNet_M(activation=act)
        name = f'M-{act}'
        print(f"\nTraining {name}")
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss()
        curves = train_one_model(model, opt, crit, train_loader, val_loader, EPOCHS, f'part1_E2_{name}')
        results_e2[name] = curves
else:
    results_e2 = {}


# ================================================================
#  E3: Different Loss Functions (w/ and w/o L2 regularization)
# ================================================================
if not SKIP_E1_E3:
    print("\n" + "=" * 50)
    print("Experiment E3: With/Without L2 Regularization (weight_decay)")
    print("=" * 50)

    results_e3 = {}
    for wd, label in [(0.0, 'No L2'), (1e-3, 'L2=1e-3'), (5e-3, 'L2=5e-3')]:
        set_seed(SEED)
        model = MyCIFARNet_M(activation='relu')
        name = f'M-{label}'
        print(f"\nTraining {name}")
        opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
        crit = nn.CrossEntropyLoss()
        curves = train_one_model(model, opt, crit, train_loader, val_loader, EPOCHS, f'part1_E3_{name}')
        results_e3[name] = curves
else:
    results_e3 = {}


# ================================================================
#  E4: Different Optimizers (Adam vs SGD vs RMSprop) [Section 1.1, Point 5(a)]
# ================================================================
print("\n" + "=" * 50)
print("Experiment E4: Different Optimizers (Adam vs SGD vs RMSprop)")
print("=" * 50)

results_e4 = {}
opt_configs = [
    ('Adam', torch.optim.Adam, {'lr': LR, 'weight_decay': 1e-4}),
    ('SGD+momentum', torch.optim.SGD, {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 1e-4}),
    ('RMSprop', torch.optim.RMSprop, {'lr': LR, 'weight_decay': 1e-4}),
]

for opt_name, opt_fn, opt_kwargs in opt_configs:
    set_seed(SEED)
    model = MyCIFARNet_M(activation='relu')
    name = f'M-{opt_name}'
    print(f"\nTraining {name}")
    opt = opt_fn(model.parameters(), **opt_kwargs)
    crit = nn.CrossEntropyLoss()
    curves = train_one_model(model, opt, crit, train_loader, val_loader, EPOCHS, f'part1_E4_{name}')
    results_e4[name] = curves


# ================================================================
#  Plot Results
# ================================================================
def plot_comparison(results_dict, title, save_name, metric_idx=3):
    """Plot validation accuracy comparison."""
    epochs = range(1, EPOCHS + 1)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    for name, (train_losses, train_accs, val_accs, _, _) in results_dict.items():
        plt.plot(epochs, train_losses, label=name, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title(f'{title} - Loss')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    for name, (train_losses, train_accs, val_accs, _, _) in results_dict.items():
        plt.plot(epochs, val_accs, label=name, linewidth=1.5)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title(f'{title} - Accuracy')
    plt.legend(fontsize=9)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(figures_path, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")


if not SKIP_E1_E3:
    plot_comparison(results_e1, 'E1: Different Filter Counts', 'part1_E1_filter_counts.png')
    plot_comparison(results_e2, 'E2: ReLU vs LeakyReLU', 'part1_E2_activations.png')
    plot_comparison(results_e3, 'E3: L2 Regularization', 'part1_E3_l2_regularization.png')
plot_comparison(results_e4, 'E4: Different Optimizers', 'part1_E4_optimizers.png')


# ================================================================
#  Summary
# ================================================================
print("\n" + "=" * 50)
print("Part 1 Summary")
print("=" * 50)
all_results = []
if not SKIP_E1_E3:
    all_results.append(("E1: Filter Counts", results_e1))
    all_results.append(("E2: Activations", results_e2))
    all_results.append(("E3: L2 Regularization", results_e3))
all_results.append(("E4: Optimizers", results_e4))

best_test_acc = 0
best_config = ""
for exp_name, rdict in all_results:
    print(f"\n{exp_name}:")
    print(f"  {'Config':<22s} {'Best Val':>10s} {'Test Acc':>10s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10}")
    for name, (_, _, _, best_val, test_acc) in rdict.items():
        print(f"  {name:<22s} {best_val:>10.4f} {test_acc:>10.4f}")
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_config = f"{exp_name}: {name}"

print(f"\n{'='*50}")
print(f"Best overall: {best_config}")
print(f"Best test accuracy: {best_test_acc:.4f} ({best_test_acc*100:.2f}%)")
print(f"Best test error: {(1-best_test_acc)*100:.2f}%")
print(f"{'='*50}")
