"""
Evaluate all saved models on CIFAR-10 test set.
Loads weights from ../reports/models/ and prints test accuracy for each.
"""
import os
import sys
import torch
from torch import nn

# Allow imports from sibling directories
sys.path.insert(0, os.path.dirname(__file__))
from data.loaders import get_cifar_loader
from models.vgg import VGG_A, VGG_A_BatchNorm
from models.my_cifar_net import MyCIFARNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Path setup ---
module_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(module_dir)
models_dir = os.path.join(parent_dir, 'reports', 'models')


def get_accuracy(model, loader):
    """Compute accuracy on a given data loader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return correct / total


def eval_model(model, weight_path, loader):
    """Load weights and evaluate."""
    state = torch.load(weight_path, map_location=DEVICE)
    # Handle both cases: saved as state_dict or saved with extra keys
    if 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    model.to(DEVICE)
    acc = get_accuracy(model, loader)
    print(f"  Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    return acc


def build_mycifarnet_from_name(filename):
    """Infer MyCIFARNet config from filename.
    
    Uses S/M/L letter or _bNN pattern to determine base_ch.
    S → base_ch=32, M → base_ch=48, L → base_ch=64.
    """
    base = os.path.splitext(filename)[0]
    kwargs = {}

    # Determine base_ch: prefer _bNN pattern, then S/M/L letter
    ch = None
    for code, val in [('_b16', 16), ('_b32', 32), ('_b48', 48), ('_b64', 64)]:
        if code in base:
            ch = val
            break
    if ch is None:
        for code, val in [('_S', 32), ('S ', 32), ('S(', 32),
                          ('_M', 48), ('M ', 48), ('M(', 48), ('M-', 48),
                          ('_L', 64), ('L ', 64), ('L(', 64), ('L-', 64)]:
            if code in base:
                ch = val
                break
    kwargs['base_ch'] = ch or 64

    # activation
    if 'leaky' in base.lower():
        kwargs['activation'] = 'leaky_relu'
    else:
        kwargs['activation'] = 'relu'

    kwargs.setdefault('dropout', 0.3)
    return MyCIFARNet(**kwargs)


if __name__ == '__main__':
    print(f"Device: {DEVICE}")
    print(f"Models dir: {models_dir}\n")

    # Load test set
    test_loader = get_cifar_loader(root='./data/', train=False, batch_size=128, shuffle=False, num_workers=0)
    print(f"Test set: {len(test_loader)} batches\n")

    # Gather all .pth files
    files = sorted(f for f in os.listdir(models_dir) if f.endswith('.pth'))

    # ======== Part 1: MyCIFARNet models ========
    print("=" * 60)
    print("Part 1: MyCIFARNet (Custom Network)")
    print("=" * 60)

    p1_files = [f for f in files if not f.startswith('VGG')]
    for fname in p1_files:
        path = os.path.join(models_dir, fname)
        print(f"\n{fname}")
        model = build_mycifarnet_from_name(fname)
        eval_model(model, path, test_loader)

    # ======== Part 2: VGG models ========
    print("\n" + "=" * 60)
    print("Part 2: VGG-A (with & without BatchNorm)")
    print("=" * 60)

    vgg_files = [f for f in files if f.startswith('VGG')]
    for fname in vgg_files:
        path = os.path.join(models_dir, fname)
        print(f"\n{fname}")
        if '_BN_' in fname:
            model = VGG_A_BatchNorm()
        else:
            model = VGG_A()
        eval_model(model, path, test_loader)

    print("\nDone.")
