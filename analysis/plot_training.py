"""
Plots training analysis from analysis/training_logs.json.
Saves all figures to analysis/.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG_PATH = "analysis/training_logs.json"
OUT_DIR = "analysis"

with open(LOG_PATH) as f:
    logs = json.load(f)

epochs      = [e["epoch"] for e in logs]
total_loss  = [e["total_loss"] for e in logs]
mlm_loss    = [e["mlm_loss"] for e in logs]
rel_loss    = [e["rel_loss"] for e in logs]
type_loss   = [e["type_loss"] for e in logs]
w_mlm       = [e["weight_mlm"] for e in logs]
w_rel       = [e["weight_rel"] for e in logs]
w_type      = [e["weight_type"] for e in logs]

# ── 1. Total loss per epoch ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, total_loss, "o-", color="#2563eb", linewidth=2, markersize=7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_title("Total Training Loss per Epoch")
ax.set_xticks(epochs)
ax.grid(True, alpha=0.3)
for x, y in zip(epochs, total_loss):
    ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_total_loss.png", dpi=150)
plt.close()
print("Saved training_total_loss.png")

# ── 2. Per-task loss per epoch (separate subplots) ───────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

task_data = [
    (mlm_loss,  "#16a34a", "o-", "MLM Loss"),
    (rel_loss,  "#dc2626", "s-", "Relation Loss"),
    (type_loss, "#9333ea", "^-", "Type Loss"),
]
for ax, (vals, color, style, title) in zip(axes, task_data):
    ax.plot(epochs, vals, style, color=color, linewidth=2, markersize=7)
    for x, y in zip(epochs, vals):
        ax.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8)
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xticks(epochs)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Epoch")
fig.suptitle("Per-Task Loss per Epoch", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_task_losses.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved training_task_losses.png")

# ── 3. Task weight evolution per epoch ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(epochs, w_mlm,  "o-", label="MLM Weight",      color="#16a34a", linewidth=2, markersize=7)
ax.plot(epochs, w_rel,  "s-", label="Relation Weight", color="#dc2626", linewidth=2, markersize=7)
ax.plot(epochs, w_type, "^-", label="Type Weight",     color="#9333ea", linewidth=2, markersize=7)
ax.set_xlabel("Epoch")
ax.set_ylabel("Softmax Weight")
ax.set_title("Dynamic Task Weight Evolution")
ax.set_xticks(epochs)
ax.legend()
ax.grid(True, alpha=0.3)
ax.axhline(1/3, color="gray", linestyle="--", alpha=0.5, label="Equal weight (1/3)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_task_weights.png", dpi=150)
plt.close()
print("Saved training_task_weights.png")

# ── 4. Step-level loss curve (all epochs combined, smoothed) ─────────────────
all_steps = []
for e in logs:
    all_steps.extend(e["step_losses"])

def smooth(y, window=50):
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="valid")

smoothed = smooth(all_steps, window=max(1, len(all_steps) // 40))
x_steps  = np.linspace(1, len(all_steps), len(smoothed))

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(range(1, len(all_steps) + 1), all_steps, color="#93c5fd", alpha=0.3, linewidth=0.5, label="Raw")
ax.plot(x_steps, smoothed, color="#2563eb", linewidth=2, label="Smoothed")

# Mark epoch boundaries
steps_per_epoch = len(all_steps) // len(epochs)
for i, ep in enumerate(epochs[:-1]):
    boundary = (i + 1) * steps_per_epoch
    ax.axvline(boundary, color="gray", linestyle="--", alpha=0.5)
    ax.text(boundary + 5, max(all_steps) * 0.95, f"Ep {ep+1}", fontsize=8, color="gray")

ax.set_xlabel("Training Step")
ax.set_ylabel("Loss")
ax.set_title("Step-level Loss Curve (all epochs)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_step_loss.png", dpi=150)
plt.close()
print("Saved training_step_loss.png")

print("\nAll training analysis plots saved to analysis/")
