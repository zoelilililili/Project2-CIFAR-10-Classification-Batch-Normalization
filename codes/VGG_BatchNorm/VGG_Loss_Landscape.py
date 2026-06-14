import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm

from models.vgg import VGG_A
from models.vgg import VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 4
batch_size = 128

module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')

device_id = device_id
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# Initialize data loader
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
for X,y in train_loader:
    print("Batch shape:", X.shape)
    print("Label shape:", y.shape)
    img = np.transpose(X[0], [1, 2, 0])
    img = img * 0.5 + 0.5
    plt.imshow(img)
    os.makedirs(figures_path, exist_ok=True)
    plt.savefig(os.path.join(figures_path, 'cifar_sample.png'))
    plt.close()
    print("Sample image saved to figures_path")
    break


def get_accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total


def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_total_grad_norm(model):
    """Compute the total L2 norm of gradients across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def train(model, optimizer, criterion, train_loader, val_loader,
          scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    learning_curve = [np.nan] * epochs_n
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    grad_norm_curve = [np.nan] * epochs_n
    max_val_accuracy = 0
    max_val_accuracy_epoch = 0

    batches_n = len(train_loader)
    losses_list = []       # per-epoch, per-batch losses
    grads = []             # per-epoch, per-batch classifier[4].weight grads
    pbar = tqdm(range(epochs_n), unit='epoch', desc='Training')
    for epoch in pbar:
        if scheduler is not None:
            scheduler.step()
        model.train()

        loss_list = []
        grad = []
        epoch_grad_norm = 0
        learning_curve[epoch] = 0

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss_list.append(loss.item())

            loss.backward()

            epoch_grad_norm += compute_total_grad_norm(model)

            try:
                g = model.classifier[4].weight.grad.clone()
                grad.append(g.cpu().numpy())
            except:
                pass

            optimizer.step()
            learning_curve[epoch] += loss.item()

        losses_list.append(loss_list)
        grads.append(grad)
        learning_curve[epoch] /= batches_n
        grad_norm_curve[epoch] = epoch_grad_norm / batches_n

        train_acc = get_accuracy(model, train_loader, device)
        val_acc = get_accuracy(model, val_loader, device)
        train_accuracy_curve[epoch] = train_acc
        val_accuracy_curve[epoch] = val_acc

        if val_acc > max_val_accuracy:
            max_val_accuracy = val_acc
            max_val_accuracy_epoch = epoch
            if best_model_path is not None:
                os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                torch.save(model.state_dict(), best_model_path)

        pbar.set_postfix({
            'loss': f'{learning_curve[epoch]:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_acc': f'{val_acc:.4f}'
        })

    return losses_list, grads, learning_curve, val_accuracy_curve, train_accuracy_curve, grad_norm_curve


def run_loss_landscape_experiment(model_class, model_name, lr_list,
                                   train_loader, val_loader, epo, device):
    """Return dicts: all_loss_curves, all_acc_curves, all_grad_norms (keyed by lr)."""
    all_loss_curves = {}
    all_acc_curves = {}
    all_grad_norms = {}
    for lr in lr_list:
        set_random_seeds(seed_value=2020, device=device)
        model = model_class()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        print(f"\nTraining {model_name} with lr={lr}")
        losses_list, grads, loss_curve, val_acc_curve, train_acc_curve, grad_norm_curve = train(
            model, optimizer, criterion, train_loader, val_loader, epochs_n=epo,
            best_model_path=os.path.join(models_path, f'{model_name}_lr{lr}_best.pth'))
        all_loss_curves[lr] = loss_curve
        all_acc_curves[lr] = val_acc_curve
        all_grad_norms[lr] = grad_norm_curve
        print(f"{model_name} lr={lr}: best val acc = {max(val_acc_curve):.4f}")
    return all_loss_curves, all_acc_curves, all_grad_norms


# ================================================================
#  Section 2.2: Training Curve Comparison (BN vs No BN, same lr)
# ================================================================
def plot_training_curves_comparison(data_no_bn, data_bn, lr, save_dir):
    """
    Plot loss and accuracy curves side-by-side for BN vs No BN under the same lr.
    data_no_bn/data_bn: dicts with keys 'loss_curve', 'acc_curve' from run_..._experiment.
    """
    loss_no_bn = data_no_bn[0][lr]  # all_loss_curves
    loss_bn = data_bn[0][lr]
    acc_no_bn = data_no_bn[1][lr]   # all_acc_curves
    acc_bn = data_bn[1][lr]

    epochs = range(1, len(loss_no_bn) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, loss_no_bn, 'r-', label='VGG-A (No BN)', linewidth=1.5)
    axes[0].plot(epochs, loss_bn, 'b-', label='VGG-A (BN)', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Training Loss')
    axes[0].set_title(f'Loss Curve (lr={lr})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, acc_no_bn, 'r-', label='VGG-A (No BN)', linewidth=1.5)
    axes[1].plot(epochs, acc_bn, 'b-', label='VGG-A (BN)', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation Accuracy')
    axes[1].set_title(f'Accuracy Curve (lr={lr})')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_curves_lr{lr}.png')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_training_curves_all_lr(data_no_bn, data_bn, lr_list, save_dir):
    """Plot training curves for all learning rates in one combined figure."""
    losses_no_bn, accs_no_bn = data_no_bn[0], data_no_bn[1]
    losses_bn, accs_bn = data_bn[0], data_bn[1]

    fig, axes = plt.subplots(len(lr_list), 1, figsize=(10, 3.5 * len(lr_list)))
    if len(lr_list) == 1:
        axes = [axes]

    for i, lr in enumerate(lr_list):
        epochs = range(1, len(losses_no_bn[lr]) + 1)
        axes[i].plot(epochs, losses_no_bn[lr], 'r-', label='No BN', linewidth=1.5)
        axes[i].plot(epochs, losses_bn[lr], 'b-', label='With BN', linewidth=1.5)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Loss')
        axes[i].set_title(f'Training Loss: BN vs No BN (lr={lr})')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_curves_all_lr.png')
    plt.savefig(save_path)
    plt.close()
    print(f"All-LR training curves saved to {save_path}")


# ================================================================
#  Section 2.3.1: Loss Landscape
# ================================================================
def compute_min_max_curves(all_curves):
    curves_array = np.array(list(all_curves.values()))  # (n_lr, n_epochs)
    min_curve = np.min(curves_array, axis=0)
    max_curve = np.max(curves_array, axis=0)
    return min_curve, max_curve


def plot_loss_landscape(curves_bn, curves_no_bn, lr_list, save_path):
    min_bn, max_bn = compute_min_max_curves(curves_bn)
    min_no_bn, max_no_bn = compute_min_max_curves(curves_no_bn)

    epochs = range(1, len(min_bn) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, min_no_bn, 'b-', label='Min', alpha=0.7)
    plt.plot(epochs, max_no_bn, 'r-', label='Max', alpha=0.7)
    plt.fill_between(epochs, min_no_bn, max_no_bn, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VGG-A (No BN) Loss Landscape')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, min_bn, 'b-', label='Min', alpha=0.7)
    plt.plot(epochs, max_bn, 'r-', label='Max', alpha=0.7)
    plt.fill_between(epochs, min_bn, max_bn, alpha=0.3)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VGG-A (With BN) Loss Landscape')
    plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Loss landscape figure saved to {save_path}")

    # Comparison on same axes
    plt.figure(figsize=(8, 6))
    curves_array_no_bn = np.array(list(curves_no_bn.values()))
    curves_array_bn = np.array(list(curves_bn.values()))
    mean_no_bn = np.mean(curves_array_no_bn, axis=0)
    mean_bn = np.mean(curves_array_bn, axis=0)
    plt.plot(epochs, mean_no_bn, 'r-', label='VGG-A (No BN) mean', linewidth=2)
    plt.fill_between(epochs, min_no_bn, max_no_bn, color='red', alpha=0.15)
    plt.plot(epochs, mean_bn, 'b-', label='VGG-A (BN) mean', linewidth=2)
    plt.fill_between(epochs, min_bn, max_bn, color='blue', alpha=0.15)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Landscape: BN vs No BN')
    plt.legend()
    comp_path = save_path.replace('.png', '_comparison.png')
    plt.savefig(comp_path)
    plt.close()
    print(f"Comparison figure saved to {comp_path}")

    return min_bn, max_bn, min_no_bn, max_no_bn


# ================================================================
#  Section 2.3.2: Gradient Predictiveness
# ================================================================
def gradient_predictiveness_analysis(model, loader, criterion, lr, device, n_steps=50):
    """
    Measure how predictive the gradient is of the nearby loss landscape.

    For a batch, compute gradient g = ∇L(θ).
    Then measure L(θ - α·g) for a range of step sizes α along -g direction.
    The first-order Taylor predicts: L(θ - α·g) ≈ L(θ) - α·||g||².
    Plot actual vs predicted to show predictiveness.
    """
    model.eval()
    data_iter = iter(loader)
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x, y = next(data_iter)
    x, y = x.to(device), y.to(device)

    # Get original params (detached) and gradient
    model.zero_grad()
    pred = model(x)
    L0 = criterion(pred, y).item()
    loss = criterion(pred, y)
    loss.backward()

    # Collect original params and grads
    orig_params = []
    grads = []
    for p in model.parameters():
        orig_params.append(p.data.clone())
        if p.grad is not None:
            grads.append(p.grad.data.clone())
        else:
            grads.append(torch.zeros_like(p.data))

    grad_norm_sq = sum((g ** 2).sum().item() for g in grads if g is not None)

    # Evaluate L(θ - α*g) for various α
    alphas = np.linspace(0, lr * 5, n_steps)
    actual_losses = []
    predicted_losses = []

    with torch.no_grad():
        for alpha in alphas:
            # θ_new = θ - α * g
            for p, o, g in zip(model.parameters(), orig_params, grads):
                p.data.copy_(o - alpha * g)

            pred_new = model(x)
            L_alpha = criterion(pred_new, y).item()
            actual_losses.append(L_alpha)

            # First-order Taylor prediction
            L_pred = L0 - alpha * grad_norm_sq
            predicted_losses.append(L_pred)

        # Restore original params
        for p, o in zip(model.parameters(), orig_params):
            p.data.copy_(o)

    return alphas, actual_losses, predicted_losses, L0


def plot_gradient_predictiveness(model_no_bn, model_bn, loader, criterion, lr, save_path):
    """Plot gradient predictiveness comparison: BN vs No BN."""
    alphas_nb, actual_nb, pred_nb, L0_nb = gradient_predictiveness_analysis(
        model_no_bn, loader, criterion, lr, device)
    alphas_bn, actual_bn, pred_bn, L0_bn = gradient_predictiveness_analysis(
        model_bn, loader, criterion, lr, device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(alphas_nb, actual_nb, 'r-', linewidth=2, label='Actual L(θ-αg)')
    axes[0].plot(alphas_nb, pred_nb, 'r--', linewidth=1.5, label='Taylor approx.')
    axes[0].axvline(x=lr, color='gray', linestyle=':', label=f'Train lr={lr}')
    axes[0].set_xlabel('Step size α')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('VGG-A (No BN) - Gradient Predictiveness')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas_bn, actual_bn, 'b-', linewidth=2, label='Actual L(θ-αg)')
    axes[1].plot(alphas_bn, pred_bn, 'b--', linewidth=1.5, label='Taylor approx.')
    axes[1].axvline(x=lr, color='gray', linestyle=':', label=f'Train lr={lr}')
    axes[1].set_xlabel('Step size α')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('VGG-A (BN) - Gradient Predictiveness')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Gradient predictiveness figure saved to {save_path}")

    # Comparison: prediction error
    fig, ax = plt.subplots(figsize=(7, 5))
    error_nb = np.abs(np.array(actual_nb) - np.array(pred_nb))
    error_bn = np.abs(np.array(actual_bn) - np.array(pred_bn))
    ax.plot(alphas_nb, error_nb, 'r-', label='VGG-A (No BN)', linewidth=1.5)
    ax.plot(alphas_bn, error_bn, 'b-', label='VGG-A (BN)', linewidth=1.5)
    ax.set_xlabel('Step size α')
    ax.set_ylabel('|Actual - Predicted|')
    ax.set_title('Gradient Predictiveness Error (BN vs No BN)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    comp_path = save_path.replace('.png', '_error.png')
    plt.savefig(comp_path)
    plt.close()
    print(f"Predictiveness error figure saved to {comp_path}")


# ================================================================
#  Section 2.3.3: Maximum Gradient Difference (Gradient Smoothness)
# ================================================================
def gradient_smoothness_analysis(model, loader, criterion, device, epsilons=[1e-4, 5e-4, 1e-3]):
    """
    Measure the local Lipschitz constant of the gradient:
    ||g(θ + δ) - g(θ)|| / ||δ|| for small random perturbations δ.

    A smaller value means a smoother gradient landscape.
    """
    model.eval()
    data_iter = iter(loader)
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x, y = next(data_iter)
    x, y = x.to(device), y.to(device)

    results = {}

    # Compute g(θ) at current params
    model.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    g_theta = [p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data)
               for p in model.parameters()]

    for eps in epsilons:
        ratios = []
        for _ in range(5):  # average over 5 random directions
            # Generate random perturbation δ, normalize to ||δ|| = eps
            perturbations = []
            total_param_norm = 0
            with torch.no_grad():
                for p in model.parameters():
                    delta = torch.randn_like(p.data)
                    perturbations.append(delta)
                    total_param_norm += (delta ** 2).sum().item()
                total_param_norm = total_param_norm ** 0.5
                scale = eps / (total_param_norm + 1e-8)
                for i, p in enumerate(model.parameters()):
                    perturbations[i] = perturbations[i] * scale

                # Apply perturbation: θ → θ + δ
                for p, delta in zip(model.parameters(), perturbations):
                    p.data.add_(delta)

            # Compute g(θ + δ)  (outside no_grad!)
            model.zero_grad()
            pred2 = model(x)
            loss2 = criterion(pred2, y)
            loss2.backward()
            g_theta_delta = [p.grad.data.clone() if p.grad is not None
                             else torch.zeros_like(p.data)
                             for p in model.parameters()]

            # Compute ||g(θ+δ) - g(θ)|| / ||δ||
            diff_norm_sq = sum(((g2 - g1) ** 2).sum().item()
                               for g1, g2 in zip(g_theta, g_theta_delta))
            diff_norm = diff_norm_sq ** 0.5
            delta_norm = sum((d ** 2).sum().item() for d in perturbations) ** 0.5
            ratio = diff_norm / (delta_norm + 1e-8)
            ratios.append(ratio)

            # Restore original params: θ + δ → θ
            with torch.no_grad():
                for p, delta in zip(model.parameters(), perturbations):
                    p.data.sub_(delta)

        results[eps] = np.mean(ratios)

    return results


def plot_gradient_smoothness(smoothness_no_bn, smoothness_bn, save_path):
    """Plot gradient smoothness (Lipschitz constant) comparison."""
    epsilons = list(smoothness_no_bn.keys())
    values_no_bn = [smoothness_no_bn[e] for e in epsilons]
    values_bn = [smoothness_bn[e] for e in epsilons]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(epsilons))
    width = 0.35
    ax.bar([i - width/2 for i in x], values_no_bn, width, label='VGG-A (No BN)',
           color='red', alpha=0.7)
    ax.bar([i + width/2 for i in x], values_bn, width, label='VGG-A (BN)',
           color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ε={e}' for e in epsilons])
    ax.set_ylabel('||g(θ+δ) - g(θ)|| / ||δ||')
    ax.set_title('Gradient Smoothness (lower = smoother)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Gradient smoothness figure saved to {save_path}")


# ================================================================
#  Main Experiment
# ================================================================
epo = 20
lr_list = [1e-3, 2e-3, 1e-4, 5e-4]
os.makedirs(models_path, exist_ok=True)
os.makedirs(figures_path, exist_ok=True)

# --- Experiment 1: VGG_A (no BN) ---
print("=" * 50)
print("Experiment 1: VGG-A (No BN)")
print("=" * 50)
data_no_bn = run_loss_landscape_experiment(
    VGG_A, 'VGG_A', lr_list, train_loader, val_loader, epo, device)
# data_no_bn = (all_loss_curves, all_acc_curves, all_grad_norms)

# --- Experiment 2: VGG_A_BatchNorm ---
print("=" * 50)
print("Experiment 2: VGG-A with BatchNorm")
print("=" * 50)
data_bn = run_loss_landscape_experiment(
    VGG_A_BatchNorm, 'VGG_A_BN', lr_list, train_loader, val_loader, epo, device)
# data_bn = (all_loss_curves, all_acc_curves, all_grad_norms)

# ================================================================
#  Section 2.2: Plot training curve comparison (BN vs No BN)
# ================================================================
print("\n" + "=" * 50)
print("Plotting Section 2.2: Training curve comparisons")
print("=" * 50)
for lr in lr_list:
    plot_training_curves_comparison(data_no_bn, data_bn, lr, figures_path)
plot_training_curves_all_lr(data_no_bn, data_bn, lr_list, figures_path)

# ================================================================
#  Section 2.3.1: Loss Landscape
# ================================================================
print("\n" + "=" * 50)
print("Plotting Section 2.3.1: Loss Landscape")
print("=" * 50)
plot_loss_landscape(
    data_bn[0], data_no_bn[0], lr_list,
    os.path.join(figures_path, 'loss_landscape.png'))

# ================================================================
#  Section 2.3.2 + 2.3.3: Gradient Analysis (using best-lr model)
# ================================================================
print("\n" + "=" * 50)
print("Running Section 2.3.2: Gradient Predictiveness")
print("=" * 50)
# Load the best model (use lr=1e-3 as representative)
criterion = nn.CrossEntropyLoss()

model_vgg = VGG_A().to(device)
best_path_no_bn = os.path.join(models_path, 'VGG_A_lr0.001_best.pth')
model_vgg.load_state_dict(torch.load(best_path_no_bn, map_location=device))

model_bn = VGG_A_BatchNorm().to(device)
best_path_bn = os.path.join(models_path, 'VGG_A_BN_lr0.001_best.pth')
model_bn.load_state_dict(torch.load(best_path_bn, map_location=device))

plot_gradient_predictiveness(
    model_vgg, model_bn, val_loader, criterion, lr=1e-3,
    save_path=os.path.join(figures_path, 'gradient_predictiveness.png'))

print("\n" + "=" * 50)
print("Running Section 2.3.3: Gradient Smoothness (Max Gradient Difference)")
print("=" * 50)
smoothness_no_bn = gradient_smoothness_analysis(
    model_vgg, val_loader, criterion, device)
smoothness_bn = gradient_smoothness_analysis(
    model_bn, val_loader, criterion, device)
print(f"Gradient smoothness (No BN): {smoothness_no_bn}")
print(f"Gradient smoothness (BN):    {smoothness_bn}")
plot_gradient_smoothness(
    smoothness_no_bn, smoothness_bn,
    os.path.join(figures_path, 'gradient_smoothness.png'))

print("\n" + "=" * 50)
print("All experiments complete!")
print("=" * 50)
