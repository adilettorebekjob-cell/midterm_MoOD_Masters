from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

run = Path("results/codet5_v2_20261027_223413")
threshold = 0.185

# Сохраняем новые результаты отдельно от исходного эксперимента.
output = run / "evaluation_threshold_0185"
output.mkdir(exist_ok=True)

rows = [
    json.loads(line)
    for line in (run / "predictions.jsonl")
    .read_text(encoding="utf-8").splitlines()
    if line.strip()
]

y = np.array([row["target"] for row in rows])
probabilities = np.array([row["probability"] for row in rows])
predictions = (probabilities >= threshold).astype(int)

result = {
    "threshold": threshold,
    "samples": len(rows),
    "accuracy": float(accuracy_score(y, predictions)),
    "precision": float(precision_score(y, predictions, zero_division=0)),
    "recall": float(recall_score(y, predictions, zero_division=0)),
    "f1": float(f1_score(y, predictions, zero_division=0)),
    "roc_auc": float(roc_auc_score(y, probabilities)),
    "confusion_matrix": confusion_matrix(
        y, predictions, labels=[0, 1]
    ).tolist(),
}
result["accuracy_requirement_met"] = result["accuracy"] >= 0.93

(output / "metrics.json").write_text(
    json.dumps(result, indent=2),
    encoding="utf-8",
)

print(json.dumps(result, indent=2))

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y, predictions,
    labels=[0, 1],
    display_labels=["Safe", "Vulnerable"],
    cmap="Blues",
    ax=ax,
)
ax.set_title(f"CodeT5 V2 — threshold {threshold}")
fig.tight_layout()
fig.savefig(output / "confusion_matrix.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(6, 5))
RocCurveDisplay.from_predictions(y, probabilities, ax=ax)
ax.set_title("CodeT5 V2 — ROC curve")
fig.tight_layout()
fig.savefig(output / "roc.png", dpi=150)
plt.show()

print("Результаты сохранены:", output.resolve())