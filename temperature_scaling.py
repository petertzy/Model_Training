import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

# Load saved logits and labels
logits = np.load("logits.npy")
labels = np.load("labels.npy")

logits_tensor = torch.tensor(logits)
labels_tensor = torch.tensor(labels)

def temperature_scale(logits, T):
    return logits / T

def nll_loss(T):
    scaled_logits = temperature_scale(logits_tensor, T)
    log_probs = F.log_softmax(scaled_logits, dim=1)
    loss = F.nll_loss(log_probs, labels_tensor)
    return loss.item()

# Find best T
res = minimize(nll_loss, x0=np.array([1.0]), bounds=[(0.5, 5.0)])
best_T = res.x[0]
print("Best temperature:", best_T)

# Example calibrated confidence
scaled_logits = logits_tensor / best_T
calibrated_probs = F.softmax(scaled_logits, dim=1)
calibrated_confidence = calibrated_probs.max(dim=1).values
print(calibrated_confidence[:10])
