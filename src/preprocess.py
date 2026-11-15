"""Veri yükleme, temizleme, öznitelik seçimi ve normalleştirme.

Desteklenen kaynaklar
---------------------
* CIC-IoT2023: ``label`` sütunu içeren CSV dosyaları.
* N-BaIoT: etiket sütunu bulunmayan, cihaz klasörlerine ayrılmış CSV dosyaları;
  etiket dosya yolundan (``benign``, ``gafgyt``, ``mirai``) türetilir.
* Sentetik yedek veri: CSV verilmezse mimariyi uçtan uca çalıştırmak için
  dengesiz, çok sınıflı sentetik ağ trafiği üretilir.

Sınıf kodları (aileler): 0: Normal, 1: Mirai, 2: BASHLITE, 3: DDoS.
Veri setinde bulunmayan aileler (ör. CIC-IoT2023'te BASHLITE) atlanır ve
etiketler ardışık olacak şekilde yeniden numaralandırılır.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

CLASS_NAMES = ["Normal", "Mirai", "BASHLITE", "DDoS"]
LABEL_COLUMN_CANDIDATES = ("label", "Label", "class", "Class", "attack", "Attack", "attack_cat")


# --------------------------------------------------------------------------- #
# Veri kümesi kapsayıcısı
# --------------------------------------------------------------------------- #
@dataclass
class Dataset:
    """Modellere verilmeye hazır, ölçeklenmiş veri bölünmesi.

    Zero-day bölünmesinde ``y_test`` *özgün* sınıf kodlarını taşır (eğitimde hiç
    görülmeyen sınıf dahil); ``y_train`` ise eğitimde bulunan sınıflar için
    ardışık yeniden kodlanmış etiketleri içerir (``train_classes`` eşlemesi).
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    class_names: List[str]
    feature_names: List[str]
    source: str
    scaler: StandardScaler
    selector: SelectKBest
    normal_index: int = 0
    train_classes: Optional[List[int]] = None
    unknown_mask: Optional[np.ndarray] = None
    held_out: Optional[str] = None

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])


def to_sequence(X: np.ndarray) -> np.ndarray:
    """(n, k) tablosunu Conv1D/RNN katmanları için (n, k, 1) dizisine çevirir."""
    return np.asarray(X, dtype=np.float32)[..., np.newaxis]


# --------------------------------------------------------------------------- #
# Yükleme
# --------------------------------------------------------------------------- #
def _family_from_text(text: str) -> Optional[int]:
    t = str(text).lower()
    if "benign" in t or "normal" in t:
        return 0
    if "mirai" in t:
        return 1
    if "gafgyt" in t or "bashlite" in t:
        return 2
    if "ddos" in t or "dos" in t:  # "ddos" ve "dos" (DoS-*) aynı aileye eşlenir
        return 3
    return None


def _normal_index(class_names: List[str]) -> int:
    for i, name in enumerate(class_names):
        low = name.lower()
        if "benign" in low or "normal" in low:
            return i
    return 0


def load_dataset(
    path: str,
    label_col: Optional[str] = None,
    family_labels: bool = True,
    max_rows_per_file: Optional[int] = 50_000,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Bir CSV dosyasını ya da klasördeki tüm CSV'leri okur.

    Etiket; ``label_col`` ya da bilinen etiket sütunu adlarından biriyle bulunur.
    Sütun yoksa (N-BaIoT) etiket, kök klasöre göreli dosya yolundan türetilir.
    ``family_labels=True`` iken ham etiketler 4 aileye indirgenir, eşlenemeyenler
    (ör. Recon, Spoofing) atılır. ``max_rows_per_file`` her dosyanın ilk N satırını
    okuyarak bellek kullanımını sınırlar.
    """
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True))
        root = path
    elif os.path.isfile(path):
        files, root = [path], os.path.dirname(path) or "."
    else:
        raise FileNotFoundError(f"Yol bulunamadı: {path}")
    if not files:
        raise FileNotFoundError(f"'{path}' altında CSV dosyası yok.")

    frames = []
    for f in files:
        df = pd.read_csv(f, nrows=max_rows_per_file, low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
        col = label_col or next((c for c in LABEL_COLUMN_CANDIDATES if c in df.columns), None)
        if col is not None and col in df.columns:
            raw = df.pop(col).astype(str).to_numpy()
        else:
            raw = np.full(len(df), os.path.relpath(f, root))
        df["__label__"] = raw
        frames.append(df)
    data = pd.concat(frames, ignore_index=True, sort=False)
    raw = data.pop("__label__")

    if family_labels:
        fam = raw.map(_family_from_text)
        keep = fam.notna().to_numpy()
        if (~keep).any():
            print(f"[preprocess] Dört aileye eşlenemeyen {int((~keep).sum()):,} satır atıldı.")
        if not keep.any():
            raise ValueError("Hiçbir etiket Normal/Mirai/BASHLITE/DDoS ailelerine eşlenemedi; "
                             "family_labels=False deneyin.")
        data = data.loc[keep].reset_index(drop=True)
        ids = fam[keep].astype(int).to_numpy()
        present, y = np.unique(ids, return_inverse=True)
        names = [CLASS_NAMES[i] for i in present]
    else:
        names = sorted(raw.unique().tolist())
        y = raw.map({n: i for i, n in enumerate(names)}).to_numpy()
    return data, np.asarray(y, dtype=np.int64), names


def generate_synthetic_traffic(
    n_samples: int = 20_000,
    n_features: int = 40,
    class_ratios: Tuple[float, ...] = (0.55, 0.10, 0.05, 0.30),
    seed: int = 42,
    corrupt_fraction: float = 3e-4,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Sınıf dengesizliği olan sentetik ağ trafiği (0: Normal, 1: Mirai, 2: BASHLITE, 3: DDoS).

    BASHLITE merkezi Mirai'ye yakın tutulur (aynı botnet ailesi benzerliği);
    bazı öznitelikler çarpık (log-normal) dağılır ve hücrelerin küçük bir kısmına
    ``inf``/``NaN`` yerleştirilir, böylece temizleme adımı gerçekten çalışır.
    """
    rng = np.random.default_rng(seed)
    ratios = np.asarray(class_ratios, dtype=float)
    ratios = ratios / ratios.sum()
    counts = np.maximum((ratios * n_samples).astype(int), 20)
    n_inf = max(8, int(n_features * 0.6))

    centers = np.zeros((4, n_features))
    centers[0, :n_inf] = rng.normal(0, 0.60, n_inf)
    centers[1, :n_inf] = rng.normal(0, 1.00, n_inf)
    centers[2, :n_inf] = centers[1, :n_inf] + rng.normal(0, 0.55, n_inf)
    centers[3, :n_inf] = rng.normal(0, 1.00, n_inf)
    scales = rng.uniform(0.7, 1.5, size=(4, n_features))
    mix = np.eye(n_features) + rng.normal(0, 0.15, (n_features, n_features))

    xs, ys = [], []
    for c in range(4):
        z = rng.normal(size=(counts[c], n_features)) * scales[c]
        xs.append(z @ mix + centers[c])
        ys.append(np.full(counts[c], c))
    X = np.vstack(xs)
    y = np.concatenate(ys)
    X[:, :8] = np.exp(0.5 * X[:, :8])  # hız/bayt oranı benzeri çarpık öznitelikler

    perm = rng.permutation(len(y))
    X, y = X[perm], y[perm]

    k = max(1, int(X.size * corrupt_fraction))
    idx = rng.choice(X.size, size=2 * k, replace=False)
    np.put(X, idx[:k], np.inf)
    np.put(X, idx[k:], np.nan)

    cols = [f"feat_{i:02d}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y.astype(np.int64), list(CLASS_NAMES)


# --------------------------------------------------------------------------- #
# Temizleme
# --------------------------------------------------------------------------- #
def clean_features(X: pd.DataFrame, y: np.ndarray) -> Tuple[pd.DataFrame, np.ndarray]:
    """Sayısal olmayan/tamamen boş sütunları, ``inf``/``NaN`` içeren satırları ve sabit sütunları atar."""
    X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    keep = X.notna().all(axis=1).to_numpy()
    n_bad = int((~keep).sum())
    X = X.loc[keep].reset_index(drop=True)
    y = np.asarray(y)[keep]
    constant = X.columns[X.nunique() <= 1]
    X = X.drop(columns=constant)
    print(f"[preprocess] Temizleme: {n_bad:,} satır (inf/NaN) ve {len(constant)} sabit sütun atıldı; "
          f"kalan: {len(X):,} satır x {X.shape[1]} öznitelik.")
    return X, y


def _compact_labels(y: np.ndarray, names: List[str]) -> Tuple[np.ndarray, List[str]]:
    present, y_new = np.unique(y, return_inverse=True)
    if len(present) == len(names):
        return y, names
    return y_new.astype(np.int64), [names[i] for i in present]


def load_raw(
    csv_path: Optional[str] = None,
    n_synthetic: int = 20_000,
    seed: int = 42,
    **load_kwargs,
) -> Tuple[pd.DataFrame, np.ndarray, List[str], str]:
    """CSV'yi yükler (verilmediyse sentetik veri üretir) ve temizler."""
    if csv_path:
        X, y, names = load_dataset(csv_path, **load_kwargs)
        source = f"csv:{csv_path}"
    else:
        print("[preprocess] CSV verilmedi -> sentetik ağ trafiği üretiliyor "
              "(0: Normal, 1: Mirai, 2: BASHLITE, 3: DDoS).")
        X, y, names = generate_synthetic_traffic(n_samples=n_synthetic, seed=seed)
        source = "synthetic"
    X, y = clean_features(X, y)
    y, names = _compact_labels(y, names)
    return X, y, names, source


# --------------------------------------------------------------------------- #
# Öznitelik seçimi + normalleştirme (yalnızca eğitim verisine fit edilir)
# --------------------------------------------------------------------------- #
def _select_and_scale(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.ndarray, k: int):
    k = min(k, X_train.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k).fit(X_train.to_numpy(), y_train)
    names = [str(c) for c in X_train.columns[selector.get_support()]]
    Xtr = selector.transform(X_train.to_numpy())
    Xte = selector.transform(X_test.to_numpy())
    scaler = StandardScaler().fit(Xtr)
    return (scaler.transform(Xtr).astype(np.float32),
            scaler.transform(Xte).astype(np.float32), scaler, selector, names)


def split_and_scale(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: List[str],
    source: str = "",
    k: int = 20,
    test_size: float = 0.2,
    seed: int = 42,
) -> Dataset:
    """Katmanlı train/test bölmesi; SelectKBest ve StandardScaler yalnızca eğitime fit edilir."""
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    Xtr, Xte, scaler, selector, names = _select_and_scale(X_tr, X_te, y_tr, k)
    return Dataset(Xtr, Xte, y_tr, y_te, list(class_names), names, source, scaler, selector,
                   normal_index=_normal_index(class_names))


def prepare_data(
    csv_path: Optional[str] = None,
    k: int = 20,
    test_size: float = 0.2,
    seed: int = 42,
    n_synthetic: int = 20_000,
    **load_kwargs,
) -> Dataset:
    """Yükle -> temizle -> böl -> SelectKBest -> StandardScaler."""
    X, y, names, source = load_raw(csv_path, n_synthetic=n_synthetic, seed=seed, **load_kwargs)
    return split_and_scale(X, y, names, source, k=k, test_size=test_size, seed=seed)


def prepare_zero_day_split(
    X: pd.DataFrame,
    y: np.ndarray,
    class_names: List[str],
    held_out_class: Union[int, str],
    k: int = 20,
    test_size: float = 0.2,
    seed: int = 42,
    source: str = "",
) -> Dataset:
    """Zero-day genellenebilirlik testi için bir saldırı sınıfını eğitimden çıkarır.

    Seçilen sınıfın *tüm* örnekleri yalnızca test kümesine taşınır. Öznitelik
    seçimi ve ölçekleyici, sınıfa dair hiçbir bilgi sızmaması için eğitim
    kümesine (sınıf çıkarıldıktan sonra) fit edilir. Dönen ``Dataset``:

    * ``y_train``: eğitimde kalan sınıflar için ardışık kodlar (``train_classes`` ile eşlenir),
    * ``y_test``: özgün sınıf kodları, ``unknown_mask``: test içinde bilinmeyen sınıfın konumu.
    """
    held = class_names.index(held_out_class) if isinstance(held_out_class, str) else int(held_out_class)
    if not 0 <= held < len(class_names):
        raise ValueError(f"Geçersiz held_out_class: {held_out_class}")
    normal = _normal_index(class_names)
    if held == normal:
        raise ValueError("Normal sınıfı zero-day saldırı olarak ayrılamaz.")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)
    move = y_tr == held
    X_te = pd.concat([X_te, X_tr.loc[move]])
    y_te = np.concatenate([y_te, y_tr[move]])
    X_tr, y_tr = X_tr.loc[~move], y_tr[~move]

    train_classes = sorted(int(c) for c in np.unique(y_tr))
    remap = {old: new for new, old in enumerate(train_classes)}
    y_tr_new = np.array([remap[int(v)] for v in y_tr], dtype=np.int64)

    Xtr, Xte, scaler, selector, names = _select_and_scale(X_tr, X_te, y_tr_new, k)
    return Dataset(
        Xtr, Xte, y_tr_new, y_te.astype(np.int64),
        [class_names[c] for c in train_classes], names, source, scaler, selector,
        normal_index=remap[normal], train_classes=train_classes,
        unknown_mask=(y_te == held), held_out=class_names[held],
    )
