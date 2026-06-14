"""
简单训练 MyCIFARNet，给定参数，输出训练曲线和测试准确率。
使用: python train_simple.py
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

from models.my_cifar_net import MyCIFARNet
from data.loaders import get_cifar_loader

# ========== 参数（在这里改） ==========
BATCH_SIZE = 128
EPOCHS = 30
LR = 0.001
WEIGHT_DECAY = 0           # L2 正则化 (实验表明不加最好)

BASE_CH = 64             
ACTIVATION = 'relu'        # 'relu' 或 'leaky_relu'
DROPOUT = 0.3
SEED = 2020

# ========== 设备和路径 ==========
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
os.makedirs(figures_path, exist_ok=True)
os.makedirs(models_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ========== 种子 ==========
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ========== 数据 ==========
print("Loading CIFAR-10...")
train_loader = get_cifar_loader(train=True, batch_size=BATCH_SIZE)
test_loader = get_cifar_loader(train=False, batch_size=BATCH_SIZE)

# ========== 模型 ==========
model = MyCIFARNet(base_ch=BASE_CH, activation=ACTIVATION, dropout=DROPOUT).to(device)
params_m = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model: base_ch={BASE_CH}, activation={ACTIVATION}, dropout={DROPOUT}, params={params_m:.2f}M")
print(f"Optimizer: Adam, lr={LR}, weight_decay={WEIGHT_DECAY}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
criterion = nn.CrossEntropyLoss()

# ========== 训练 ==========
train_losses = []
test_accs = []
best_acc = 0
model_name = f"MyCIFARNet_b{BASE_CH}_{ACTIVATION}_wd{WEIGHT_DECAY}"

pbar = tqdm(range(EPOCHS), unit='epoch')
for epoch in pbar:
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    train_losses.append(total_loss / len(train_loader))

    # 测试
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            _, predicted = torch.max(pred, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    acc = correct / total
    test_accs.append(acc)

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(),
                   os.path.join(models_path, f'{model_name}_best.pth'))

    pbar.set_postfix({'loss': f'{train_losses[-1]:.4f}', 'test_acc': f'{acc:.4f}'})

# ========== 最终测试（用最佳权重） ==========
model.load_state_dict(torch.load(os.path.join(models_path, f'{model_name}_best.pth')))
model.eval()
correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        _, predicted = torch.max(model(x), 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
test_acc = correct / total

print(f"\n{'='*40}")
print(f"  Best test accuracy : {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"  Test error         : {(1-test_acc)*100:.2f}%")
print(f"  Model saved to     : models/{model_name}_best.pth")
print(f"{'='*40}")

# ========== 画图 ==========
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(range(1, EPOCHS+1), train_losses, 'b-')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].set_title('Training Loss'); axes[0].grid(True, alpha=0.3)
axes[1].plot(range(1, EPOCHS+1), test_accs, 'g-')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_title('Test Accuracy'); axes[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(figures_path, f'{model_name}_result.png'))
plt.close()
print(f"Figure saved to figures/{model_name}_result.png")
