import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# ---------------------------------------------------------
# IMPORT MODEL + DATASET
# ---------------------------------------------------------

from models.swin.swin_model import SwinHABSegmentation
import importlib.util

dataset_path = os.path.join(
    PROJECT_ROOT,
    "scripts",
    "74_create_pytorch_dataset.py"
)

spec = importlib.util.spec_from_file_location(
    "dataset_module",
    dataset_path
)

dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)

MultiWaterbodyHABDataset = dataset_module.MultiWaterbodyHABDataset

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

SPLIT_CSV = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed",
    "train_val_split",
    "multilocation_train_val_split.csv"
)

CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "models",
    "swin",
    "multilocation_corrected_best_swin_hab_model.pth"
)

RESULT_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "swin",
    "step82_diagnosis"
)

os.makedirs(RESULT_DIR, exist_ok=True)

# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device("cpu")

print("=" * 70)
print("STEP 82 - CORRECTED SWIN MODEL DIAGNOSIS")
print("=" * 70)

print("\nDevice:", device)
print("Checkpoint:", CHECKPOINT)

# ---------------------------------------------------------
# LOAD SPLIT
# ---------------------------------------------------------

split_df = pd.read_csv(SPLIT_CSV)

val_df = split_df[
    split_df["split"].str.lower() == "val"
].copy()

print("\nValidation tiles:", len(val_df))

# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

val_dataset = MultiWaterbodyHABDataset(
    val_df,
    os.path.join(PROJECT_ROOT, "data", "tiles", "images"),
    os.path.join(PROJECT_ROOT, "data", "tiles", "masks")
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = SwinHABSegmentation(
    num_channels=14,
    num_classes=2
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

# Handle either state_dict directly or checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

print("Model loaded successfully.")

# ---------------------------------------------------------
# STORAGE
# ---------------------------------------------------------

all_true = []
all_pred = []

waterbody_results = {}

# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------

with torch.no_grad():

    for batch_idx, (images, masks) in enumerate(val_loader):

        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        true_mask = masks[0].cpu().numpy()
        pred_mask = predictions[0].cpu().numpy()

        # Valid pixels
        valid = true_mask != 255

        true_valid = true_mask[valid].astype(np.uint8)
        pred_valid = pred_mask[valid].astype(np.uint8)

        all_true.extend(true_valid.tolist())
        all_pred.extend(pred_valid.tolist())

        # Get tile information
        row = val_df.iloc[batch_idx]

        tile_id = str(row["tile_id"])

        # Try to determine waterbody
        if "waterbody" in row.index:
            waterbody = str(row["waterbody"])
        else:
            waterbody = tile_id.split("_")[0]

        if waterbody not in waterbody_results:
            waterbody_results[waterbody] = {
                "true": [],
                "pred": [],
                "tiles": 0
            }

        waterbody_results[waterbody]["true"].extend(
            true_valid.tolist()
        )

        waterbody_results[waterbody]["pred"].extend(
            pred_valid.tolist()
        )

        waterbody_results[waterbody]["tiles"] += 1

        if (batch_idx + 1) % 25 == 0:
            print(
                f"Processed validation tiles: "
                f"{batch_idx + 1}/{len(val_dataset)}"
            )

# ---------------------------------------------------------
# OVERALL METRICS
# ---------------------------------------------------------

all_true = np.array(all_true)
all_pred = np.array(all_pred)

precision = precision_score(
    all_true,
    all_pred,
    zero_division=0
)

recall = recall_score(
    all_true,
    all_pred,
    zero_division=0
)

f1 = f1_score(
    all_true,
    all_pred,
    zero_division=0
)

iou = jaccard_score(
    all_true,
    all_pred,
    zero_division=0
)

actual_hab = np.mean(all_true == 1) * 100
predicted_hab = np.mean(all_pred == 1) * 100

print("\n" + "=" * 70)
print("OVERALL VALIDATION RESULTS")
print("=" * 70)

print(f"Precision       : {precision:.6f}")
print(f"Recall          : {recall:.6f}")
print(f"F1 Score        : {f1:.6f}")
print(f"IoU             : {iou:.6f}")
print(f"Actual HAB %    : {actual_hab:.2f}%")
print(f"Predicted HAB % : {predicted_hab:.2f}%")

# ---------------------------------------------------------
# WATERBODY-WISE RESULTS
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("WATERBODY-WISE RESULTS")
print("=" * 70)

rows = []

for waterbody, data in sorted(waterbody_results.items()):

    true_arr = np.array(data["true"])
    pred_arr = np.array(data["pred"])

    wb_precision = precision_score(
        true_arr,
        pred_arr,
        zero_division=0
    )

    wb_recall = recall_score(
        true_arr,
        pred_arr,
        zero_division=0
    )

    wb_f1 = f1_score(
        true_arr,
        pred_arr,
        zero_division=0
    )

    wb_iou = jaccard_score(
        true_arr,
        pred_arr,
        zero_division=0
    )

    wb_actual = np.mean(true_arr == 1) * 100
    wb_predicted = np.mean(pred_arr == 1) * 100

    print(f"\n{waterbody}")
    print("-" * 50)
    print(f"Validation tiles : {data['tiles']}")
    print(f"Precision         : {wb_precision:.6f}")
    print(f"Recall            : {wb_recall:.6f}")
    print(f"F1                : {wb_f1:.6f}")
    print(f"IoU               : {wb_iou:.6f}")
    print(f"Actual HAB %      : {wb_actual:.2f}%")
    print(f"Predicted HAB %   : {wb_predicted:.2f}%")

    rows.append({
        "waterbody": waterbody,
        "validation_tiles": data["tiles"],
        "precision": wb_precision,
        "recall": wb_recall,
        "f1": wb_f1,
        "iou": wb_iou,
        "actual_hab_percent": wb_actual,
        "predicted_hab_percent": wb_predicted
    })

# ---------------------------------------------------------
# SAVE REPORT
# ---------------------------------------------------------

report_df = pd.DataFrame(rows)

report_path = os.path.join(
    RESULT_DIR,
    "waterbody_wise_results.csv"
)

report_df.to_csv(
    report_path,
    index=False
)

print("\nSaved waterbody-wise report:")
print(report_path)

# ---------------------------------------------------------
# SAVE OVERALL SUMMARY
# ---------------------------------------------------------

summary_df = pd.DataFrame([{
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "iou": iou,
    "actual_hab_percent": actual_hab,
    "predicted_hab_percent": predicted_hab
}])

summary_path = os.path.join(
    RESULT_DIR,
    "overall_results.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)

print("\nSaved overall report:")
print(summary_path)

print("\n" + "=" * 70)
print("STEP 82 COMPLETE")
print("=" * 70)