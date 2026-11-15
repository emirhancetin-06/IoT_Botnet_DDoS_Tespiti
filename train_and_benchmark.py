#!/usr/bin/env python3
"""Uçtan uca eğitim, kıyaslama ve uç cihaz optimizasyonu.

Akış: veri hazırlama -> sınıf ağırlıkları -> klasik ML + derin öğrenme modellerinin eğitimi
-> metrik tablosu (konsol + CSV) -> karmaşıklık matrisi -> (isteğe bağlı) zero-day testi
-> Lightweight_CNN_LSTM için budama + TFLite nicemleme ve çıkarım raporu.

Örnekler::

    python train_and_benchmark.py                       # sentetik veri, tüm modeller
    python train_and_benchmark.py --data /veri/CIC-IoT2023 --k 30
    python train_and_benchmark.py --zero-day --zero-day-class BASHLITE
    python train_and_benchmark.py --skip-dl             # TensorFlow olmadan yalnızca klasik modeller
"""
from __future__ import annotations

import argparse
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

from src import preprocess as pp

CLASSICAL_MODELS = ["RandomForest", "XGBoost"]
DL_MODELS = ["CNN", "LSTM", "GRU", "Lightweight_CNN_LSTM"]
PROPOSED = "Lightweight_CNN_LSTM"


# --------------------------------------------------------------------------- #
# Yardımcılar
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=None, help="CSV dosyası ya da klasörü (CIC-IoT2023 / N-BaIoT). "
                                                "Verilmezse sentetik veri üretilir.")
    p.add_argument("--label-col", default=None, help="Etiket sütunu adı (varsayılan: otomatik).")
    p.add_argument("--max-rows-per-file", type=int, default=50_000, help="Dosya başına okunacak en çok satır.")
    p.add_argument("--n-synthetic", type=int, default=20_000, help="Sentetik örnek sayısı.")
    p.add_argument("--k", type=int, default=20, help="SelectKBest ile seçilecek öznitelik sayısı.")
    p.add_argument("--epochs", type=int, default=15, help="Derin modeller için en çok epoch (EarlyStopping açık).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results")
    p.add_argument("--skip-dl", action="store_true",
                   help="Derin öğrenme ve uç optimizasyonunu atla (TensorFlow gerekmez).")
    p.add_argument("--zero-day", action="store_true", help="Zero-day genellenebilirlik testini de çalıştır.")
    p.add_argument("--zero-day-class", default="BASHLITE",
                   help="Eğitimden çıkarılacak saldırı sınıfı (ad ya da kod; varsayılan: BASHLITE).")
    return p.parse_args()


def compute_class_weights(y: np.ndarray, n_classes: int) -> dict:
    classes = np.arange(n_classes)
    weights = compute_class_weight("balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, weights)}


def compute_metrics(y_true, y_pred, proba, n_classes: int) -> dict:
    """Makro ortalamalı metrikler. FPR: sınıf-başına (bire-karşı-hepsi) FP/(FP+TN) ortalaması."""
    labels = list(range(n_classes))
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fp = cm.sum(axis=0) - np.diag(cm)
    tn = cm.sum() - (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm))
    fpr = float(np.mean(fp / np.maximum(fp + tn, 1)))
    proba = np.asarray(proba, dtype=np.float64)
    proba = proba / proba.sum(axis=1, keepdims=True)
    auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    return {"Accuracy": accuracy_score(y_true, y_pred), "Precision": p, "Recall": r,
            "F1-Score": f1, "FPR": fpr, "ROC-AUC": auc}


def load_dl():
    """TensorFlow bileşenlerini geç yükler; yoksa anlaşılır bir hatayla çıkar."""
    try:
        from tensorflow import keras
        from src.models import MODEL_BUILDERS
    except ImportError as exc:
        sys.exit(f"HATA: TensorFlow yüklenemedi ({exc}).\n"
                 "  - Kurulum: pip install -r requirements.txt\n"
                 "  - Yalnızca klasik modeller için: python train_and_benchmark.py --skip-dl")
    return keras, MODEL_BUILDERS


def fit_predict(name: str, ds: pp.Dataset, cw: dict, args, dl):
    """Modeli eğitir; (olasılıklar, eğitim süresi sn, Keras modeli | None) döndürür."""
    t0 = time.perf_counter()
    model = None
    if name == "RandomForest":
        clf = RandomForestClassifier(n_estimators=150, class_weight=cw, n_jobs=-1, random_state=args.seed)
        clf.fit(ds.X_train, ds.y_train)
        proba = clf.predict_proba(ds.X_test)
    elif name == "XGBoost":
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, tree_method="hist",
                            objective="multi:softprob", eval_metric="mlogloss",
                            n_jobs=-1, random_state=args.seed)
        sample_weight = np.array([cw[int(c)] for c in ds.y_train])
        clf.fit(ds.X_train, ds.y_train, sample_weight=sample_weight)
        proba = clf.predict_proba(ds.X_test)
    else:
        keras, builders = dl
        keras.utils.set_random_seed(args.seed)
        Xtr, Xte = pp.to_sequence(ds.X_train), pp.to_sequence(ds.X_test)
        model = builders[name](Xtr.shape[1:], ds.n_classes)
        model.fit(Xtr, ds.y_train, validation_split=0.1, epochs=args.epochs, batch_size=args.batch_size,
                  class_weight=cw, verbose=0,
                  callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=4,
                                                           restore_best_weights=True)])
        proba = model.predict(Xte, batch_size=1024, verbose=0)
    return proba, time.perf_counter() - t0, model


def plot_confusion_matrix(y_true, y_pred, class_names, title: str, path: str) -> None:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Örnek sayısı")
    sns.heatmap(cm_norm, annot=True, fmt=".1%", cmap="Blues", cbar=False, ax=axes[1],
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    axes[1].set_title("Satır bazında normalize (duyarlılık)")
    for ax in axes:
        ax.set_xlabel("Tahmin edilen sınıf")
        ax.set_ylabel("Gerçek sınıf")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_table(df: pd.DataFrame) -> str:
    shown = df.copy()
    for col in ["Accuracy", "Precision", "Recall", "F1-Score", "FPR", "ROC-AUC"]:
        shown[col] = df[col].map(lambda v: f"{v:.4f}")
    shown["Train_s"] = df["Train_s"].map(lambda v: f"{v:.1f}")
    shown["Params"] = df["Params"].map(lambda v: "-" if pd.isna(v) else f"{int(v):,}")
    return shown.to_string(index=False)


# --------------------------------------------------------------------------- #
# Zero-day testi
# --------------------------------------------------------------------------- #
def run_zero_day(args, raw, models, dl, out_dir: str) -> None:
    X, y, names, source = raw
    held = int(args.zero_day_class) if str(args.zero_day_class).isdigit() else args.zero_day_class
    ds = pp.prepare_zero_day_split(X, y, names, held, k=args.k, seed=args.seed, source=source)
    cw = compute_class_weights(ds.y_train, ds.n_classes)
    print(f"\n[zero-day] '{ds.held_out}' eğitimden çıkarıldı: eğitim sınıfları = {ds.class_names}; "
          f"testteki bilinmeyen örnek sayısı = {int(ds.unknown_mask.sum()):,}")

    normal_orig = ds.train_classes[ds.normal_index]
    y_attack = ds.y_test != normal_orig
    rows = []
    for name in models:
        proba, _, _ = fit_predict(name, ds, cw, args, dl)
        pred = np.asarray(ds.train_classes)[proba.argmax(axis=1)]
        p_attack = pred != normal_orig
        known_attack = y_attack & ~ds.unknown_mask
        rows.append({
            "Model": name,
            "Held_out": ds.held_out,
            "Unknown_Detection_Rate": float(p_attack[ds.unknown_mask].mean()),
            "Known_Attack_Recall": float(p_attack[known_attack].mean()),
            "Binary_F1": f1_score(y_attack, p_attack),
            "Binary_FPR": float(p_attack[~y_attack].mean()),
        })
    df = pd.DataFrame(rows)
    print("\n[zero-day] Bilinmeyen saldırının 'saldırı' olarak yakalanma oranı "
          "(Normal dışı herhangi bir sınıfa atanması):")
    print(df.round(4).to_string(index=False))
    df.round(4).to_csv(os.path.join(out_dir, "zero_day_results.csv"), index=False)


# --------------------------------------------------------------------------- #
# Ana akış
# --------------------------------------------------------------------------- #
def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    dl = None if args.skip_dl else load_dl()
    models = []
    for name in CLASSICAL_MODELS:
        if name == "XGBoost":
            try:
                import xgboost  # noqa: F401
            except ImportError:
                print("UYARI: xgboost yüklü değil; XGBoost atlanıyor (pip install xgboost).")
                continue
        models.append(name)
    if dl:
        models += DL_MODELS

    raw = pp.load_raw(args.data, n_synthetic=args.n_synthetic, seed=args.seed,
                      label_col=args.label_col, max_rows_per_file=args.max_rows_per_file)
    X, y, names, source = raw
    ds = pp.split_and_scale(X, y, names, source, k=args.k, seed=args.seed)
    cw = compute_class_weights(ds.y_train, ds.n_classes)

    counts = pd.Series(ds.y_train).value_counts().sort_index()
    print(f"\nVeri kaynağı: {ds.source} | eğitim: {len(ds.X_train):,} | test: {len(ds.X_test):,} | "
          f"öznitelik: {ds.n_features}")
    print("Eğitim sınıf dağılımı ve 'balanced' sınıf ağırlıkları:")
    for c in range(ds.n_classes):
        print(f"  {c} {ds.class_names[c]:<9} n={counts.get(c, 0):>7,}  ağırlık={cw[c]:.3f}")

    rows, preds, trained = [], {}, {}
    for name in models:
        print(f"\n>>> {name} eğitiliyor...")
        proba, secs, model = fit_predict(name, ds, cw, args, dl)
        pred = proba.argmax(axis=1)
        preds[name] = pred
        if model is not None:
            trained[name] = model
        rows.append({"Model": name, **compute_metrics(ds.y_test, pred, proba, ds.n_classes),
                     "Params": model.count_params() if model is not None else np.nan,
                     "Train_s": secs})
        print(f"    tamamlandı ({secs:.1f} sn) | Accuracy={rows[-1]['Accuracy']:.4f} | F1={rows[-1]['F1-Score']:.4f}")

    results = pd.DataFrame(rows)
    print("\n" + "=" * 100 + "\n KIYASLAMA SONUÇLARI (makro ortalama; test kümesi)\n" + "=" * 100)
    print(format_table(results))
    csv_path = os.path.join(args.output_dir, "benchmark_results.csv")
    results.round(4).to_csv(csv_path, index=False)

    best = PROPOSED if PROPOSED in preds else results.loc[results["F1-Score"].idxmax(), "Model"]
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(ds.y_test, preds[best], ds.class_names, f"Karmaşıklık matrisi - {best}", cm_path)
    print(f"\nKaydedildi: {csv_path}\nKaydedildi: {cm_path} ({best})")

    if args.zero_day:
        run_zero_day(args, raw, models, dl, args.output_dir)

    if PROPOSED in trained:
        from src import edge_optimizer
        edge_optimizer.run_edge_pipeline(
            trained[PROPOSED], pp.to_sequence(ds.X_train), ds.y_train,
            pp.to_sequence(ds.X_test), ds.y_test, args.output_dir,
            class_weight=cw, batch_size=args.batch_size)
    else:
        print("\n[edge] Uç cihaz optimizasyonu atlandı (derin öğrenme modelleri çalıştırılmadı).")


if __name__ == "__main__":
    main()
