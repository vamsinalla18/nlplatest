"""
Runs 5 training epochs purely for logging/visualization.
Does NOT overwrite model.pt — logs saved to analysis/training_logs.json.
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from training.trainer import Trainer

print("Starting training analysis run (5 epochs, model.pt will NOT be overwritten)...")
trainer = Trainer()
trainer.train(
    epochs=5,
    save_model=False,
    log_path="analysis/training_logs.json",
)
print("Logs saved to analysis/training_logs.json")
