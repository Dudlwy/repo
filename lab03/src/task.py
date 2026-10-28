import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchinfo import summary
import thop
import numpy as np
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

torch.manual_seed(42)

# --- 1. Dataset & DataLoaders ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])

full_train = datasets.FashionMNIST(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root=DATA_DIR, train=False, download=True, transform=transform)

train_dataset, val_dataset = random_split(full_train, [55000, 5000], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.net(x)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )
    def forward(self, x):
        return self.net(x)

class ViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_proj = nn.Linear(49, 64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        # x: (B, 1, 28, 28) -> Patches (B, 16, 49)
        B = x.shape[0]
        x = x.unfold(2, 7, 7).unfold(3, 7, 7)
        x = x.contiguous().view(B, 1, 16, 7, 7).squeeze(1).view(B, 16, 49)
        x = self.patch_proj(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

models = {'MLP': MLP(), 'CNN': CNN(), 'Attention': ViT()}


def measure_latency(model, batch_size, runs=100):
    model.eval()
    dummy = torch.randn(batch_size, 1, 28, 28)
    with torch.no_grad():
        model(dummy) # Warmup
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(dummy)
            times.append(time.perf_counter() - t0)
    return np.median(times) * 1000

print("=== ARCHITECTURE COST & PROFILING ===")
metrics = {}

for name, model in models.items():
    dummy = torch.randn(1, 1, 28, 28)
    flops, params = thop.profile(model, inputs=(dummy,), verbose=False)
    
    lat_1 = measure_latency(model, batch_size=1, runs=100)
    lat_128 = measure_latency(model, batch_size=128, runs=20)
    
    metrics[name] = {
        'params': params,
        'flops': flops,
        'lat_1': lat_1,
        'lat_128': lat_128
    }
    print(f"[{name}] Params: {params:,.0f} | THOP FLOPs: {flops:,.0f} | Latency B1: {lat_1:.3f}ms | Latency B128: {lat_128:.3f}ms")


print("\n=== TRAINING (5 Epochs) ===")
test_accuracies = {}

for name, model in models.items():
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        model.train()
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_b, y_b in test_loader:
            preds = torch.argmax(model(X_b), dim=1)
            correct += (preds == y_b).sum().item()
            total += y_b.size(0)
            
    acc = (correct / total) * 100
    test_accuracies[name] = acc
    print(f"[{name}] Test Accuracy: {acc:.2f}%")


names = list(models.keys())
accs = [test_accuracies[n] for n in names]
params_list = [metrics[n]['params'] for n in names]
lats_b1 = [metrics[n]['lat_1'] for n in names]

# Accuracy vs Params
plt.figure(figsize=(6, 4))
plt.scatter(params_list, accs, color='blue', s=100)
for i, txt in enumerate(names):
    plt.annotate(txt, (params_list[i], accs[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlabel("Parameters")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Parameter Count")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_params.png"))
plt.close()


plt.figure(figsize=(6, 4))
plt.scatter(lats_b1, accs, color='red', s=100)
for i, txt in enumerate(names):
    plt.annotate(txt, (lats_b1[i], accs[i]), textcoords="offset points", xytext=(0, 10), ha='center')
plt.xlabel("Latency B=1 (ms)")
plt.ylabel("Test Accuracy (%)")
plt.title("Accuracy vs Inference Latency")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_vs_latency.png"))
plt.close()

print("\nPlots saved to results/ directory.")