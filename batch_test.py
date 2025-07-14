import pandas as pd
import requests
import time
import numpy as np
from scipy.optimize import minimize
import torch
import torch.nn.functional as F

API_URL = "http://0.0.0.0:8000/predict"

logits_np = np.load("logits.npy")
labels_np = np.load("labels.npy")
logits_tensor = torch.tensor(logits_np)
labels_tensor = torch.tensor(labels_np)

def temperature_scale(logits, T):
    return logits / T

def nll_loss(T):
    scaled_logits = temperature_scale(logits_tensor, T)
    log_probs = F.log_softmax(scaled_logits, dim=1)
    loss = F.nll_loss(log_probs, labels_tensor)
    return loss.item()

res = minimize(nll_loss, x0=np.array([1.0]), bounds=[(0.5, 5.0)])
best_T = res.x[0]
print("Best temperature found:", best_T)

def calibrate_logits(logits, T):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    scaled_logits = logits_tensor / T
    calibrated_probs = F.softmax(scaled_logits, dim=0).numpy()
    return calibrated_probs

predicted_labels = []
confidences = []

df = pd.read_csv("test.csv")

for i, row in df.iterrows():
    article = row["Article"]
    comment = row["Comment"]

    payload = {
        "Article": article,
        "Comment": comment
    }

    try:
        response = requests.post(API_URL, json=payload)
        response.raise_for_status()
        result = response.json()
        print("API raw response:", result)

        logits = result["Logits"]
        calibrated_probs = calibrate_logits(logits, best_T)
        calibrated_confidence = float(np.max(calibrated_probs))
        predicted_label = int(np.argmax(calibrated_probs))

        predicted_labels.append(predicted_label)
        confidences.append(calibrated_confidence)

        print(f"Row {i} done - label={predicted_label}, confidence={calibrated_confidence:.4f}")

    except Exception as e:
        print(f"Error at row {i}: {e}")
        predicted_labels.append(None)
        confidences.append(None)

    time.sleep(0.1)

df["predicted_label"] = predicted_labels
df["predicted_confidence"] = confidences

valid_rows = df[df["predicted_label"].notnull()]
total = len(valid_rows)
correct = (valid_rows["Relevance"] == valid_rows["predicted_label"]).sum()

if total > 0:
    accuracy = correct / total
    print(f"\n Accuracy: {accuracy:.4%} ({correct}/{total})")
else:
    print("\n No valid predictions available to calculate accuracy.")

# Save accuracy as first row in CSV
csv_file = "test_with_predictions.csv"

if total > 0:
    accuracy_percent = f"{accuracy:.2%}"
    header_line = f"Accuracy: {accuracy_percent}"
else:
    header_line = "Accuracy: N/A"

# Save header + data
with open(csv_file, "w", encoding="utf-8") as f:
    f.write(header_line + "\n")
    df.to_csv(f, index=False)

print(f"Done! Predictions saved to {csv_file}")

