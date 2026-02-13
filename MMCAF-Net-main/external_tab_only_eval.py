import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    log_loss,
)
from sklearn.preprocessing import StandardScaler
import pandas as pd

import models


def _infer_num_tab_features_from_state_dict(state_dict):
    for k in [
        "module.kan.scale_base",
        "kan.scale_base",
        "module.kan.mask",
        "kan.mask",
    ]:
        if k in state_dict:
            t = state_dict[k]
            if hasattr(t, "shape") and len(t.shape) == 2:
                return int(t.shape[0])
    raise ValueError("Could not infer num_tab_features from checkpoint state_dict.")


def _load_model_from_ckpt(ckpt_path, device, gpu_ids):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model_name = ckpt.get("model_name", "MMCAF_Net")
    state = ckpt.get("model_state", ckpt)
    num_tab_features = _infer_num_tab_features_from_state_dict(state)
    model_fn = models.__dict__.get(model_name, models.__dict__.get("MMCAF_Net"))
    model = model_fn(num_tab_features=num_tab_features)
    if gpu_ids:
        model = nn.DataParallel(model, gpu_ids)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if len(unexpected) > 0:
        print(f"WARNING: Unexpected keys in ckpt: {unexpected[:5]}")
    if len(missing) > 0:
        print(f"WARNING: Missing keys in ckpt: {missing[:5]}")
    model = model.to(device)
    model.eval()
    return model, num_tab_features


def _fit_scaler_from_internal_csv(data_dir, feature_cols):
    csv_path = os.path.join(data_dir, "G_first_last_nor.csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    x = df[feature_cols].copy().fillna(0.0).to_numpy(dtype=np.float32, copy=True)
    scaler = StandardScaler()
    scaler.fit(x)
    return scaler


def _load_external_excel(excel_path, feature_cols, label_col="STAS", scaler=None):
    df = pd.read_excel(excel_path)
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in external excel.")
    x = df[feature_cols].copy().fillna(0.0).to_numpy(dtype=np.float32, copy=True)
    if scaler is not None:
        x = scaler.transform(x).astype(np.float32, copy=False)
    y = df[label_col].to_numpy()
    y = pd.to_numeric(y, errors="coerce").fillna(0).to_numpy(dtype=np.float32, copy=False)
    y = (y > 0.5).astype(np.int64)
    return x, y


def evaluate_tab_only(model, device, x, y, batch_size=64, use_amp=False):
    logits_all = []
    y_all = []
    enabled_amp = bool(use_amp and torch.cuda.is_available())
    amp_dtype = torch.bfloat16 if (enabled_amp and torch.cuda.is_bf16_supported()) else torch.float16
    with torch.inference_mode():
        for start in range(0, len(x), batch_size):
            end = min(start + batch_size, len(x))
            xb = torch.tensor(x[start:end], dtype=torch.float32, device=device)
            yb = torch.tensor(y[start:end], dtype=torch.float32, device=device).unsqueeze(1)
            img = torch.zeros((xb.size(0), 1, 12, 192, 192), dtype=torch.float32, device=device)
            with torch.cuda.amp.autocast(enabled=enabled_amp, dtype=amp_dtype):
                out = model.forward(img, xb, mode=2)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(out, yb, reduction="mean")
            logits_all.append(out.detach().cpu().numpy().reshape(-1))
            y_all.append(y[start:end])
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(y_all, axis=0).astype(np.int64)
    probs = 1.0 / (1.0 + np.exp(-logits))

    metrics = {}
    metrics["loss"] = float(log_loss(labels, probs, labels=[0, 1]))
    preds = (probs >= 0.5).astype(np.int64)
    metrics["Accuracy"] = float(accuracy_score(labels, preds))
    metrics["F1"] = float(f1_score(labels, preds, zero_division=0))
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics["Sensitivity"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    metrics["Specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    has_valid = len(np.unique(labels)) > 1
    metrics["AUROC"] = float(roc_auc_score(labels, probs)) if has_valid else float("nan")
    metrics["AUPRC"] = float(average_precision_score(labels, probs)) if has_valid else float("nan")
    curves = {}
    if has_valid:
        curves["ROC"] = roc_curve(labels, probs, pos_label=1)
        curves["PRC"] = precision_recall_curve(labels, probs, pos_label=1)
    curves["Confusion Matrix"] = cm
    return metrics, curves


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--external_excel", type=str, required=True)
    p.add_argument("--data_dir", type=str, default="../data")
    p.add_argument("--label_col", type=str, default="STAS")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gpu_ids", type=str, default="0")
    p.add_argument("--use_amp", type=lambda s: str(s).lower() in ["1", "true", "yes"], default=False)
    args = p.parse_args()

    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() and len(gpu_ids) > 0 else "cpu")

    desired_feature_cols = ["实性成分大小", "毛刺征", "支气管异常征", "胸膜凹陷征", "CEA"]
    scaler = _fit_scaler_from_internal_csv(args.data_dir, desired_feature_cols)
    x, y = _load_external_excel(args.external_excel, desired_feature_cols, label_col=args.label_col, scaler=scaler)
    print(f"External samples: {len(y)}, positives: {int(y.sum())}, negatives: {int((1-y).sum())}")

    model, num_tab_features = _load_model_from_ckpt(args.ckpt_path, device, gpu_ids)
    if num_tab_features != x.shape[1]:
        raise ValueError(f"Feature dim mismatch: ckpt expects {num_tab_features}, excel provides {x.shape[1]}")

    metrics, _ = evaluate_tab_only(model, device, x, y, batch_size=args.batch_size, use_amp=args.use_amp)
    print("External TabOnly Metrics:")
    for k in ["loss", "AUROC", "AUPRC", "Accuracy", "F1", "Sensitivity", "Specificity"]:
        v = metrics.get(k, None)
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            print(f"  {k}: n/a")
        else:
            print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

