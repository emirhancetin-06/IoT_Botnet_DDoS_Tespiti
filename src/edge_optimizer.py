"""Uç cihaz (Raspberry Pi sınıfı) optimizasyonu: budama + TFLite dinamik aralık nicemleme.

Budama, Keras 3 ile uyumluluk sorunlarından kaçınmak ve LSTM ağırlıklarını da
kapsamak için bu modülde uygulanan, kademeli (polinom takvimli) büyüklük tabanlı
bir yöntemdir: ağırlıkların mutlak değerce en küçük kısmı sıfırlanır, maskeler ince
ayar boyunca korunur. Çıkış katmanı budanmaz.

Ölçümler ana makinede, tek iş parçacığıyla yapılır; Raspberry Pi için bir
*simülasyondur*. Gerçek cihazdaki süreler daha yüksek olacaktır.
"""
from __future__ import annotations

import gc
import gzip
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import tensorflow as tf
from tensorflow import keras

layers = keras.layers


# --------------------------------------------------------------------------- #
# Budama
# --------------------------------------------------------------------------- #
def _prunable_layers(model: keras.Model) -> List[Tuple[keras.layers.Layer, Tuple[int, ...]]]:
    """(katman, budanacak ağırlık indeksleri) listesi; çıkış katmanı hariç."""
    out = []
    for layer in model.layers[:-1]:
        if isinstance(layer, (layers.LSTM, layers.GRU)):
            out.append((layer, (0, 1)))  # kernel + recurrent_kernel
        elif isinstance(layer, (layers.Conv1D, layers.Dense)):
            out.append((layer, (0,)))  # kernel
    return out


class MagnitudePruningCallback(keras.callbacks.Callback):
    """Polinom (kübik) seyreklik takvimiyle kademeli büyüklük tabanlı budama."""

    def __init__(self, final_sparsity: float = 0.5, initial_sparsity: float = 0.0,
                 total_steps: int = 100, update_every: int = 10):
        super().__init__()
        self.final_sparsity = final_sparsity
        self.initial_sparsity = initial_sparsity
        self.total_steps = total_steps
        self.update_every = update_every
        self._step = 0
        self._masks: Dict[Tuple[str, int], np.ndarray] = {}

    def _target(self) -> float:
        end = max(int(0.8 * self.total_steps), 1)  # son %20: seyreklik sabit, yalnızca ince ayar
        frac = min(self._step / end, 1.0)
        return self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * (1.0 - frac) ** 3

    def _apply(self, sparsity: float) -> None:
        for layer, idxs in _prunable_layers(self.model):
            weights = layer.get_weights()
            for i in idxs:
                w = weights[i]
                key = (layer.name, i)
                mask = self._masks.get(key, np.ones(w.shape, dtype=bool))
                if sparsity > 0:
                    mask = mask & (np.abs(w) > np.quantile(np.abs(w), sparsity))
                self._masks[key] = mask
                weights[i] = w * mask
            layer.set_weights(weights)

    def on_train_batch_end(self, batch, logs=None):
        self._step += 1
        if self._step % self.update_every == 0:
            self._apply(self._target())

    def on_train_end(self, logs=None):
        self._apply(self.final_sparsity)


def measure_sparsity(model: keras.Model) -> float:
    """Budanabilir ağırlıklardaki sıfır oranı (0-1)."""
    zeros = total = 0
    for layer, idxs in _prunable_layers(model):
        weights = layer.get_weights()
        for i in idxs:
            zeros += int(np.sum(weights[i] == 0))
            total += weights[i].size
    return zeros / total if total else 0.0


def prune_model(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                class_weight: Optional[Dict[int, float]] = None, final_sparsity: float = 0.5,
                epochs: int = 3, batch_size: int = 256, learning_rate: float = 5e-4
                ) -> Tuple[keras.Model, float]:
    """Modelin bir kopyasını budar ve ince ayar yapar; (budanmış model, seyreklik) döndürür."""
    pruned = keras.models.clone_model(model)
    pruned.set_weights(model.get_weights())
    pruned.compile(optimizer=keras.optimizers.Adam(learning_rate),
                   loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    steps = epochs * math.ceil(len(X_train) / batch_size)
    callback = MagnitudePruningCallback(final_sparsity=final_sparsity, total_steps=steps)
    pruned.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
               class_weight=class_weight, callbacks=[callback], verbose=0)
    return pruned, measure_sparsity(pruned)


# --------------------------------------------------------------------------- #
# TFLite dönüştürme
# --------------------------------------------------------------------------- #
def convert_to_tflite(model: keras.Model, quantize: bool = True) -> Tuple[bytes, str]:
    """Keras modelini TFLite'a çevirir. ``quantize=True``: dinamik aralık nicemleme (int8 ağırlık).

    Önce yalnızca yerleşik (builtin) operatörler denenir; LSTM bu şekilde
    dönüştürülemezse SELECT_TF_OPS ile yeniden denenir ve kullanılan mod bildirilir.
    """

    def _convert(use_select_ops: bool) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        if quantize:
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
        if use_select_ops:
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                                   tf.lite.OpsSet.SELECT_TF_OPS]
            converter._experimental_lower_tensor_list_ops = False
        return converter.convert()

    try:
        return _convert(False), "TFLITE_BUILTINS"
    except Exception as exc:  # noqa: BLE001
        print(f"[edge] UYARI: yalnızca yerleşik operatörlerle dönüştürme başarısız "
              f"({type(exc).__name__}); SELECT_TF_OPS ile yeniden deneniyor.")
        return _convert(True), "TFLITE_BUILTINS + SELECT_TF_OPS"


def compress_to_tflite(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                       output_path: str, class_weight: Optional[Dict[int, float]] = None,
                       final_sparsity: float = 0.5, finetune_epochs: int = 3,
                       batch_size: int = 256) -> Dict[str, object]:
    """Budama + TFLite dinamik aralık nicemleme uygular ve ``output_path`` dosyasına yazar.

    Dönen sözlük: yol, seyreklik, op modu, float32 TFLite boyutu, sıkıştırılmış boyut ve
    gzip boyutu (budamanın depolama kazancı gzip ile görünür; TFLite seyrek depolamaz).
    """
    pruned, sparsity = prune_model(model, X_train, y_train, class_weight=class_weight,
                                   final_sparsity=final_sparsity, epochs=finetune_epochs,
                                   batch_size=batch_size)
    float_blob, _ = convert_to_tflite(model, quantize=False)
    blob, ops = convert_to_tflite(pruned, quantize=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as fh:
        fh.write(blob)
    return {
        "path": output_path,
        "pruned_model": pruned,
        "sparsity": sparsity,
        "ops": ops,
        "float32_tflite_kb": len(float_blob) / 1024,
        "compressed_tflite_kb": len(blob) / 1024,
        "compressed_gzip_kb": len(gzip.compress(blob)) / 1024,
    }


# --------------------------------------------------------------------------- #
# Uç cihaz performans ölçümü
# --------------------------------------------------------------------------- #
def benchmark_tflite(tflite_path: str, X: np.ndarray, y: Optional[np.ndarray] = None,
                     n_warmup: int = 50, n_runs: int = 1000) -> Dict[str, float]:
    """``.tflite`` dosyasının boyutunu, paket başına çıkarım süresini ve RAM kullanımını ölçer.

    * Süre: tek örnek (batch=1) için set_tensor + invoke + get_tensor, ms cinsinden.
    * RAM: yorumlayıcı oluşturulmadan önceki ve çıkarım sırasındaki en yüksek RSS farkı (MB).
      Tüm süreç RSS'i ölçüldüğü için sonuç gürültülü olabilir; yaklaşık değer olarak okunmalıdır.
    """
    X = np.ascontiguousarray(X, dtype=np.float32)
    n = min(n_runs, len(X))
    gc.collect()
    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss
    peak = rss_before

    interpreter = tf.lite.Interpreter(model_path=tflite_path, num_threads=1)
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]

    def _infer(i: int) -> np.ndarray:
        interpreter.set_tensor(inp["index"], X[i:i + 1])
        interpreter.invoke()
        return interpreter.get_tensor(out["index"])

    for i in range(min(n_warmup, n)):
        _infer(i)

    times = np.empty(n)
    preds = np.empty(n, dtype=np.int64)
    for i in range(n):
        t0 = time.perf_counter()
        probs = _infer(i)
        times[i] = (time.perf_counter() - t0) * 1000.0
        preds[i] = int(np.argmax(probs))
        if i % 100 == 0:
            peak = max(peak, proc.memory_info().rss)
    peak = max(peak, proc.memory_info().rss)

    result = {
        "tflite_size_kb": os.path.getsize(tflite_path) / 1024,
        "avg_inference_ms": float(times.mean()),
        "p95_inference_ms": float(np.percentile(times, 95)),
        "throughput_pps": float(1000.0 / times.mean()),
        "ram_mb": max(peak - rss_before, 0) / (1024 * 1024),
        "n_runs": int(n),
    }
    if y is not None:
        result["tflite_accuracy"] = float((preds == np.asarray(y)[:n]).mean())
    return result


def print_edge_report(report: Dict[str, object]) -> None:
    line = "=" * 64
    print(f"\n{line}\n UÇ CİHAZ OPTİMİZASYON RAPORU (Raspberry Pi simülasyonu)\n{line}")
    print(f" Budama seyrekliği            : {report['sparsity']:.1%}")
    print(f" TFLite operatör modu         : {report['ops']}")
    print(f" Float32 TFLite boyutu        : {report['float32_tflite_kb']:.2f} KB")
    print(f" Budanmış + nicemlenmiş boyut : {report['tflite_size_kb']:.2f} KB")
    print(f" gzip ile sıkıştırılmış boyut : {report['compressed_gzip_kb']:.2f} KB")
    print(f" Paket başına çıkarım süresi  : {report['avg_inference_ms']:.4f} ms "
          f"(p95: {report['p95_inference_ms']:.4f} ms, ~{report['throughput_pps']:.0f} paket/sn)")
    print(f" Bellek (RAM) tüketimi        : {report['ram_mb']:.2f} MB")
    if "keras_accuracy" in report:
        print(f" Doğruluk (Keras -> TFLite)   : {report['keras_accuracy']:.4f} -> "
              f"{report['tflite_accuracy']:.4f}  (ilk {report['n_runs']} test örneği)")
    print(f" Model dosyası                : {report['path']}\n{line}")


def run_edge_pipeline(model: keras.Model, X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray, output_dir: str,
                      class_weight: Optional[Dict[int, float]] = None,
                      final_sparsity: float = 0.5, finetune_epochs: int = 3,
                      batch_size: int = 256) -> Dict[str, object]:
    """Sıkıştır -> kaydet -> ölç -> raporla -> ``edge_report.csv`` olarak yaz.

    ``X_*`` dizileri (n, k, 1) biçiminde olmalıdır.
    """
    path = os.path.join(output_dir, "lightweight_cnn_lstm_pruned_dynq.tflite")
    info = compress_to_tflite(model, X_train, y_train, path, class_weight=class_weight,
                              final_sparsity=final_sparsity, finetune_epochs=finetune_epochs,
                              batch_size=batch_size)
    report = {**{k: v for k, v in info.items() if k != "pruned_model"},
              **benchmark_tflite(path, X_test, y_test)}

    n = report["n_runs"]
    keras_pred = np.argmax(model.predict(X_test[:n], batch_size=1024, verbose=0), axis=1)
    report["keras_accuracy"] = float((keras_pred == np.asarray(y_test)[:n]).mean())

    print_edge_report(report)
    pd.DataFrame([report]).round(4).to_csv(os.path.join(output_dir, "edge_report.csv"), index=False)
    return report
