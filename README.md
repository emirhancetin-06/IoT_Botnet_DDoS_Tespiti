# IoT Ağlarında Derin Öğrenme Tabanlı Botnet ve DDoS Saldırılarının Tespiti: Hafif Bir CNN-LSTM Modelinin Tasarımı ve Değerlendirilmesi

Gazi Üniversitesi TUSAŞ-Kazan Meslek Yüksekokulu kapsamında yürütülen akademik projenin kaynak kodu ve yeniden üretilebilir deney altyapısıdır.

| | |
|---|---|




| **Araştırmacılar** | Emirhan Çetin, Oğuz Kağan Altunbaş |
| **Danışman** | Dr. Öğr. Üyesi Kubilay Ayturan |
| **Kurum** | Gazi Üniversitesi TUSAŞ-Kazan MYO |

## 1. Özet

Kaynak kısıtlı IoT cihazlarında çalışabilecek, ağ trafiğindeki botnet (Mirai, BASHLITE) ve DDoS saldırılarını tespit eden hafif bir hibrit **1B-evrişim + LSTM** modeli önerilmektedir. Evrişim katmanları akış özniteliklerindeki yerel örüntüleri, LSTM katmanı ise öznitelikler arası bağımlılıkları öğrenir. Model; tekil CNN, LSTM, GRU ile klasik makine öğrenmesi yöntemleri (Random Forest, XGBoost) karşısında Accuracy, Precision, Recall, F1-Score, FPR ve ROC-AUC ölçütleriyle kıyaslanır. Eğitilen model budama ve TensorFlow Lite dinamik aralık nicemleme ile sıkıştırılarak model boyutu, paket başına çıkarım süresi ve bellek tüketimi ölçülür. Bilinmeyen (zero-day) bir saldırı türüne genellenebilirlik ayrıca test edilebilir.

## 2. Proje Mimarisi

```
.
├── train_and_benchmark.py     # Ana çalıştırma dosyası (eğitim, kıyaslama, rapor)
├── requirements.txt
├── src/
│   ├── preprocess.py          # Yükleme, temizleme, SelectKBest, StandardScaler, sentetik veri, zero-day bölmesi
│   ├── models.py              # Lightweight_CNN_LSTM, Standalone_CNN / LSTM / GRU
│   └── edge_optimizer.py      # Budama, TFLite nicemleme, boyut/süre/RAM ölçümü
├── data/                      # CSV veri setlerinin konulacağı klasör (git'e eklenmez)
└── results/                   # benchmark_results.csv, confusion_matrix.png, ...
```

**Veri hattı.** CSV yükleme (CIC-IoT2023: `label` sütunu; N-BaIoT: etiket dosya yolundan) → `inf`/`NaN` ve sabit sütunların temizlenmesi → katmanlı train/test bölmesi → `SelectKBest(f_classif)` → `StandardScaler`. Seçici ve ölçekleyici **yalnızca eğitim verisine** fit edilir (veri sızıntısı önlenir). Etiketler dört aileye indirgenir: 0 Normal, 1 Mirai, 2 BASHLITE, 3 DDoS; veri setinde bulunmayan aileler atlanır. CSV verilmezse sınıf dengesizliği içeren sentetik trafik üretilir.

**Önerilen model (`Lightweight_CNN_LSTM`, ≈20 bin parametre).**
`Conv1D(32) → BatchNorm → MaxPool → Dropout → Conv1D(64) → BatchNorm → MaxPool → Dropout → LSTM(32) → Dense(32) → Dropout → Softmax`

**Sınıf dengesizliği.** `compute_class_weight('balanced')` ile hesaplanan ağırlıklar derin modellere `class_weight`, Random Forest'a `class_weight`, XGBoost'a `sample_weight` olarak uygulanır.

**Uç cihaz optimizasyonu.** Kademeli (polinom takvimli) büyüklük tabanlı budama (varsayılan %50 seyreklik, LSTM ağırlıkları dahil, çıkış katmanı hariç) ve ince ayar; ardından TFLite dinamik aralık nicemleme (int8 ağırlık). Raporlanan değerler: dosya boyutu (KB), gzip boyutu, paket başına ortalama çıkarım süresi (ms, tek iş parçacığı, batch=1) ve RAM tüketimi (MB). Ölçümler ana makinede yapılan bir **Raspberry Pi simülasyonudur**; gerçek cihazda süreler daha yüksek olacaktır, nihai değerler için betik cihaz üzerinde çalıştırılmalıdır. Not: TFLite seyrek depolama yapmadığından budamanın depolama kazancı yalnızca gzip boyutunda görünür.

**Metrik tanımları.** Precision, Recall ve F1 makro ortalamadır. FPR, sınıf-başına (bire-karşı-hepsi) `FP/(FP+TN)` değerlerinin ortalamasıdır. ROC-AUC bire-karşı-hepsi makro ortalamadır.

## 3. Kurulum

```bash
git clone <depo-adresi>
cd <depo-klasörü>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Kullanım

```bash
# Sentetik veriyle uçtan uca çalıştırma (veri indirmeye gerek yok)
python train_and_benchmark.py

# Gerçek veri seti (CSV dosyası ya da klasör)
python train_and_benchmark.py --data data/CIC-IoT2023 --k 30 --epochs 20

# Zero-day testi: BASHLITE eğitimden çıkarılır, yalnızca testte bulunur
python train_and_benchmark.py --zero-day --zero-day-class BASHLITE

# TensorFlow olmadan yalnızca klasik modeller
python train_and_benchmark.py --skip-dl
```

Üretilen çıktılar `results/` altına yazılır: `benchmark_results.csv`, `confusion_matrix.png`, (varsa) `zero_day_results.csv`, `edge_report.csv` ve `lightweight_cnn_lstm_pruned_dynq.tflite`.

## 5. Örnek Çalıştırma Çıktısı

Aşağıdaki çıktı `python train_and_benchmark.py --skip-dl` komutuyla (sentetik veri, yalnızca Random Forest) alınmıştır. Derin öğrenme modelleri ve uç cihaz raporu için komutun parametresiz çalıştırılması gerekir; bu çalıştırma `results/` içindeki dosyaların üzerine yazar.

```
UYARI: xgboost yüklü değil; XGBoost atlanıyor (pip install xgboost).
[preprocess] CSV verilmedi -> sentetik ağ trafiği üretiliyor (0: Normal, 1: Mirai, 2: BASHLITE, 3: DDoS).
[preprocess] Temizleme: 471 satır (inf/NaN) ve 0 sabit sütun atıldı; kalan: 19,529 satır x 40 öznitelik.

Veri kaynağı: synthetic | eğitim: 15,623 | test: 3,906 | öznitelik: 20
Eğitim sınıf dağılımı ve 'balanced' sınıf ağırlıkları:
  0 Normal    n=  8,594  ağırlık=0.454
  1 Mirai     n=  1,562  ağırlık=2.500
  2 BASHLITE  n=    782  ağırlık=4.995
  3 DDoS      n=  4,685  ağırlık=0.834

====================================================================================================
 KIYASLAMA SONUÇLARI (makro ortalama; test kümesi)
====================================================================================================
       Model Accuracy Precision Recall F1-Score    FPR ROC-AUC Params Train_s
RandomForest   0.8666    0.7591 0.7103   0.7249 0.0539  0.9583      -     2.5
```

Sentetik veri, mimarinin hatasız çalıştığını göstermek içindir; Mirai ve BASHLITE sınıfları bilinçli olarak birbirine benzetilmiştir. Bu sayılar akademik bulgu değildir; makaledeki sonuçlar gerçek CIC-IoT2023 ve N-BaIoT verileriyle üretilmelidir.

## 6. Veri Setleri

* **CIC-IoT2023** — Canadian Institute for Cybersecurity. CSV dosyalarını `data/` altına koyun.
* **N-BaIoT** — Meidan vd., UCI Machine Learning Repository. Cihaz klasör yapısı korunmalıdır; etiketler `benign`, `gafgyt`, `mirai` dosya yollarından türetilir.

`--max-rows-per-file` (varsayılan 50.000) bellek kullanımını sınırlamak için her dosyanın yalnızca ilk satırlarını okur.
