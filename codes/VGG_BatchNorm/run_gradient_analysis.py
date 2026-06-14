"""
只运行 Section 2.3.2 & 2.3.3 梯度分析（跳过训练，用已有模型权重）。
使用方式: python run_gradient_analysis.py
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 加载数据
val_loader = get_cifar_loader(train=False)
criterion = nn.CrossEntropyLoss()

# 加载最佳模型
model_vgg = VGG_A().to(device)
best_path_no_bn = os.path.join(models_path, 'VGG_A_lr0.0005_best.pth')
model_vgg.load_state_dict(torch.load(best_path_no_bn, map_location=device))
print("Loaded VGG_A (No BN) from", best_path_no_bn)

model_bn = VGG_A_BatchNorm().to(device)
best_path_bn = os.path.join(models_path, 'VGG_A_BN_lr0.001_best.pth')
model_bn.load_state_dict(torch.load(best_path_bn, map_location=device))
print("Loaded VGG_A_BN from", best_path_bn)


# ---- 2.3.2: Gradient Predictiveness ----
def gradient_predictiveness_analysis(model, loader, criterion, lr, device, n_steps=50):
    model.eval()
    data_iter = iter(loader)
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x, y = next(data_iter)
    x, y = x.to(device), y.to(device)

    model.zero_grad()
    pred = model(x)
    L0 = criterion(pred, y).item()
    loss = criterion(pred, y)
    loss.backward()

    orig_params = [p.data.clone() for p in model.parameters()]
    grads = [p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data)
             for p in model.parameters()]
    grad_norm_sq = sum((g ** 2).sum().item() for g in grads if g is not None)

    alphas = np.linspace(0, lr * 5, n_steps)
    actual_losses = []
    predicted_losses = []

    with torch.no_grad():
        for alpha in alphas:
            for p, o, g in zip(model.parameters(), orig_params, grads):
                p.data.copy_(o - alpha * g)
            pred_new = model(x)
            L_alpha = criterion(pred_new, y).item()
            actual_losses.append(L_alpha)
            predicted_losses.append(L0 - alpha * grad_norm_sq)

        for p, o in zip(model.parameters(), orig_params):
            p.data.copy_(o)

    return alphas, actual_losses, predicted_losses, L0


def plot_gradient_predictiveness(model_no_bn, model_bn, loader, criterion, lr, save_path):
    alphas_nb, actual_nb, pred_nb, L0_nb = gradient_predictiveness_analysis(
        model_no_bn, loader, criterion, lr, device)
    alphas_bn, actual_bn, pred_bn, L0_bn = gradient_predictiveness_analysis(
        model_bn, loader, criterion, lr, device)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(alphas_nb, actual_nb, 'r-', linewidth=2, label='Actual L(θ-αg)')
    axes[0].plot(alphas_nb, pred_nb, 'r--', linewidth=1.5, label='Taylor approx.')
    axes[0].axvline(x=lr, color='gray', linestyle=':', label=f'Train lr={lr}')
    axes[0].set_xlabel('Step size α'); axes[0].set_ylabel('Loss')
    axes[0].set_title('VGG-A (No BN) - Gradient Predictiveness')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].plot(alphas_bn, actual_bn, 'b-', linewidth=2, label='Actual L(θ-αg)')
    axes[1].plot(alphas_bn, pred_bn, 'b--', linewidth=1.5, label='Taylor approx.')
    axes[1].axvline(x=lr, color='gray', linestyle=':', label=f'Train lr={lr}')
    axes[1].set_xlabel('Step size α'); axes[1].set_ylabel('Loss')
    axes[1].set_title('VGG-A (BN) - Gradient Predictiveness')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"Saved {save_path}")

    fig, ax = plt.subplots(figsize=(7, 5))
    error_nb = np.abs(np.array(actual_nb) - np.array(pred_nb))
    error_bn = np.abs(np.array(actual_bn) - np.array(pred_bn))
    ax.plot(alphas_nb, error_nb, 'r-', label='VGG-A (No BN)', linewidth=1.5)
    ax.plot(alphas_bn, error_bn, 'b-', label='VGG-A (BN)', linewidth=1.5)
    ax.set_xlabel('Step size α'); ax.set_ylabel('|Actual - Predicted|')
    ax.set_title('Gradient Predictiveness Error (BN vs No BN)')
    ax.legend(); ax.grid(True, alpha=0.3)
    error_path = save_path.replace('.png', '_error.png')
    plt.savefig(error_path); plt.close()
    print(f"Saved {error_path}")


# ---- 2.3.3: Gradient Smoothness ----
def gradient_smoothness_analysis(model, loader, criterion, device, epsilons=[1e-4, 5e-4, 1e-3]):
    model.eval()
    data_iter = iter(loader)
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(loader)
        x, y = next(data_iter)
    x, y = x.to(device), y.to(device)

    results = {}
    model.zero_grad()
    pred = model(x)
    loss = criterion(pred, y)
    loss.backward()
    g_theta = [p.grad.data.clone() if p.grad is not None else torch.zeros_like(p.data)
               for p in model.parameters()]

    for eps in epsilons:
        ratios = []
        for _ in range(5):
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
                for p, delta in zip(model.parameters(), perturbations):
                    p.data.add_(delta)

            model.zero_grad()
            pred2 = model(x)
            loss2 = criterion(pred2, y)
            loss2.backward()
            g_theta_delta = [p.grad.data.clone() if p.grad is not None
                             else torch.zeros_like(p.data)
                             for p in model.parameters()]

            diff_norm_sq = sum(((g2 - g1) ** 2).sum().item()
                               for g1, g2 in zip(g_theta, g_theta_delta))
            diff_norm = diff_norm_sq ** 0.5
            delta_norm = sum((d ** 2).sum().item() for d in perturbations) ** 0.5
            ratio = diff_norm / (delta_norm + 1e-8)
            ratios.append(ratio)

            with torch.no_grad():
                for p, delta in zip(model.parameters(), perturbations):
                    p.data.sub_(delta)

        results[eps] = np.mean(ratios)

    return results


def plot_gradient_smoothness(smoothness_no_bn, smoothness_bn, save_path):
    epsilons = list(smoothness_no_bn.keys())
    values_no_bn = [smoothness_no_bn[e] for e in epsilons]
    values_bn = [smoothness_bn[e] for e in epsilons]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = range(len(epsilons))
    width = 0.35
    ax.bar([i - width/2 for i in x], values_no_bn, width, label='VGG-A (No BN)', color='red', alpha=0.7)
    ax.bar([i + width/2 for i in x], values_bn, width, label='VGG-A (BN)', color='blue', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f'ε={e}' for e in epsilons])
    ax.set_ylabel('||g(θ+δ) - g(θ)|| / ||δ||')
    ax.set_title('Gradient Smoothness (lower = smoother)')
    ax.legend(); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path); plt.close()
    print(f"Saved {save_path}")


# ---- Run ----
print("\nRunning Section 2.3.2: Gradient Predictiveness...")
plot_gradient_predictiveness(
    model_vgg, model_bn, val_loader, criterion, lr=1e-3,
    save_path=os.path.join(figures_path, 'gradient_predictiveness.png'))

print("\nRunning Section 2.3.3: Gradient Smoothness...")
smoothness_no_bn = gradient_smoothness_analysis(model_vgg, val_loader, criterion, device)
smoothness_bn = gradient_smoothness_analysis(model_bn, val_loader, criterion, device)
print(f"Gradient smoothness (No BN): {smoothness_no_bn}")
print(f"Gradient smoothness (BN):    {smoothness_bn}")
plot_gradient_smoothness(
    smoothness_no_bn, smoothness_bn,
    os.path.join(figures_path, 'gradient_smoothness.png'))

print("\nDone!")
