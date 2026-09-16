#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standalone figure generator for SR Studio v21 AI screening exports.

Input: AI_Screening_Ranked.xlsx (preferred), or CSV exported from the screening result.
Output (separate figures):
  01_ROC_Curve.png/.pdf
  02_Precision_Recall_Curve.png/.pdf
  03_Confusion_Matrix.png/.pdf
  04_Screening_Efficiency.png/.pdf
  AI_Screening_Performance_Summary.csv

Usage:
  python AI_screening_figures.py
  python AI_screening_figures.py AI_Screening_Ranked.xlsx
  python AI_screening_figures.py AI_Screening_Ranked.xlsx -o figures --show
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix,
    recall_score, precision_score, f1_score, accuracy_score,
)

REQUIRED = {"Human_Label_Normalized", "CV_Probability"}


def choose_file() -> Path | None:
    """Open a native file picker when no path is supplied; gracefully fall back."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        name = filedialog.askopenfilename(
            title="Select AI screening result",
            filetypes=[("Excel/CSV", "*.xlsx *.xls *.csv"), ("All files", "*.*")],
        )
        root.destroy()
        return Path(name) if name else None
    except Exception:
        return None


def load_result(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in {".xlsx", ".xls"}:
        book = pd.ExcelFile(path)
        # SR Studio v21 export normally uses AI_Screening_Ranked.
        preferred = "AI_Screening_Ranked" if "AI_Screening_Ranked" in book.sheet_names else book.sheet_names[0]
        df = pd.read_excel(path, sheet_name=preferred)
    elif ext == ".csv":
        # utf-8-sig is the app's CSV export; fallback handles other common encodings.
        try:
            df = pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="cp949")
    else:
        raise ValueError("Input must be .xlsx, .xls, or .csv")

    df.columns = [str(c).strip() for c in df.columns]
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            "This is not an SR Studio v21 supervised AI-screening result. "
            f"Missing column(s): {', '.join(sorted(missing))}\n"
            "Use the 'AI 선별 결과 다운로드' file (AI_Screening_Ranked.xlsx)."
        )
    return df


def cv_data(df: pd.DataFrame):
    y = pd.to_numeric(df["Human_Label_Normalized"], errors="coerce")
    p = pd.to_numeric(df["CV_Probability"], errors="coerce")
    mask = y.isin([0, 1]) & p.notna() & np.isfinite(p)
    y = y.loc[mask].astype(int).to_numpy()
    p = p.loc[mask].astype(float).to_numpy()
    if len(y) == 0:
        raise ValueError("No valid cross-validation rows were found.")
    if len(np.unique(y)) < 2:
        raise ValueError("ROC/PR curves require both Include (1) and Exclude (0) labels.")
    return y, p, mask


def cv_predictions(df: pd.DataFrame, mask, p: np.ndarray) -> np.ndarray:
    """Prefer the exact CV_Prediction saved by SR Studio; infer only if absent/incomplete."""
    if "CV_Prediction" in df.columns:
        pred = pd.to_numeric(df.loc[mask, "CV_Prediction"], errors="coerce").to_numpy()
        if len(pred) == len(p) and np.isfinite(pred).all() and set(np.unique(pred)).issubset({0, 1}):
            return pred.astype(int)
    # A missing prediction column cannot reproduce the app's optimized threshold exactly.
    # 0.5 is used only as a transparent fallback.
    print("WARNING: valid CV_Prediction not found; confusion matrix uses threshold 0.5.", file=sys.stderr)
    return (p >= 0.5).astype(int)


def save(fig, outdir: Path, stem: str):
    fig.savefig(outdir / f"{stem}.png", dpi=600, bbox_inches="tight")
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description="Create separate AI screening performance figures from SR Studio v21 output.")
    ap.add_argument("input", nargs="?", help="AI_Screening_Ranked.xlsx or compatible CSV")
    ap.add_argument("-o", "--output", help="Output directory (default: <input>_figures)")
    ap.add_argument("--show", action="store_true", help="Display figures after saving")
    args = ap.parse_args()

    path = Path(args.input).expanduser() if args.input else choose_file()
    if path is None:
        # final fallback: uniquely identifiable export in current directory
        candidates = list(Path.cwd().glob("AI_Screening_Ranked*.xlsx")) + list(Path.cwd().glob("AI_Screening_Ranked*.csv"))
        if len(candidates) == 1:
            path = candidates[0]
        else:
            raise SystemExit("No input selected. Run: python AI_screening_figures.py AI_Screening_Ranked.xlsx")
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    df = load_result(path)
    y, prob, mask = cv_data(df)
    pred = cv_predictions(df, mask, prob)

    outdir = Path(args.output).expanduser() if args.output else path.parent / f"{path.stem}_figures"
    outdir.mkdir(parents=True, exist_ok=True)

    auc = roc_auc_score(y, prob)
    ap_score = average_precision_score(y, prob)
    fpr, tpr, _ = roc_curve(y, prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y, prob)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prevalence = float(y.mean())

    # 1) ROC curve
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(fpr, tpr, linewidth=2.2, label=f"AI model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, label="Random classifier")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate (Recall)", title="ROC Curve", xlim=(0,1), ylim=(0,1))
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); save(fig, outdir, "01_ROC_Curve")

    # 2) Precision-recall curve
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(recall_curve, precision_curve, linewidth=2.2, label=f"AI model (AP = {ap_score:.3f})")
    ax.axhline(prevalence, linestyle="--", linewidth=1.2, label=f"Include prevalence = {prevalence:.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–Recall Curve", xlim=(0,1), ylim=(0,1))
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(); save(fig, outdir, "02_Precision_Recall_Curve")

    # 3) Confusion matrix -- exact saved CV_Prediction when available
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(cm, cmap="Blues")
    labels = np.array([[f"TN\n{tn}", f"FP\n{fp}"], [f"FN\n{fn}", f"TP\n{tp}"]])
    threshold_text = (cm.max() + cm.min()) / 2 if cm.size else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, labels[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > threshold_text else "black")
    ax.set_xticks([0,1], ["Predicted Exclude", "Predicted Include"])
    ax.set_yticks([0,1], ["Actual Exclude", "Actual Include"])
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("AI prediction"); ax.set_ylabel("Human label")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); save(fig, outdir, "03_Confusion_Matrix")

    # 4) Screening efficiency -- same definition as app.py: safe-exclude / all records
    if "AI_Recommendation" not in df.columns:
        print("WARNING: AI_Recommendation missing; Screening Efficiency figure skipped.", file=sys.stderr)
        safe_n = review_n = total_n = 0
    else:
        rec = df["AI_Recommendation"].fillna("").astype(str).str.strip()
        total_n = len(df)
        safe_n = int(rec.eq("안전 제외 후보").sum())
        review_n = total_n - safe_n
        safe_pct = 100 * safe_n / total_n if total_n else 0
        review_pct = 100 * review_n / total_n if total_n else 0
        fig, ax = plt.subplots(figsize=(6.2, 5.2))
        bars = ax.bar(["Human review", "AI safe-exclude"], [review_pct, safe_pct])
        for bar, n, pct in zip(bars, [review_n, safe_n], [review_pct, safe_pct]):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, f"{n:,}\n({pct:.1f}%)", ha="center", va="bottom")
        ax.set_ylabel("Proportion of total records (%)")
        ax.set_title("Screening Efficiency")
        ax.set_ylim(0, max(100, max(review_pct, safe_pct) + 12))
        fig.tight_layout(); save(fig, outdir, "04_Screening_Efficiency")

    summary = pd.DataFrame([{
        "CV_labeled_n": len(y), "CV_include_n": int(y.sum()), "CV_exclude_n": int((y==0).sum()),
        "Recall": recall_score(y, pred, zero_division=0),
        "Precision": precision_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0), "Accuracy": accuracy_score(y, pred),
        "ROC_AUC": auc, "Average_Precision": ap_score,
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Total_records": total_n, "Human_review_n": review_n, "AI_safe_exclude_n": safe_n,
        "Screening_burden_reduction": (safe_n/total_n if total_n else np.nan),
    }])
    summary.to_csv(outdir / "AI_Screening_Performance_Summary.csv", index=False, encoding="utf-8-sig")

    print(f"Done. Figures saved to:\n{outdir.resolve()}")
    print(f"ROC-AUC={auc:.3f} | AP={ap_score:.3f} | Recall={summary.loc[0,'Recall']:.3f} | FN={fn}")
    if args.show:
        plt.show()
    else:
        plt.close("all")

if __name__ == "__main__":
    main()
