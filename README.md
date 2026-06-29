# Anomaly Detection Industriale con CNN-Autoencoder (MVTec AD)

## 📌 Obiettivo del progetto

L'obiettivo del progetto è sviluppare un sistema in grado di riconoscere **difetti industriali mai osservati durante l'addestramento**.

Invece di utilizzare un approccio supervisionato, basato su esempi di prodotti sani e difettosi, viene adottato un approccio **unsupervised** basato sulla ricostruzione dell'immagine:

[
X -> X'
]

Il modello viene addestrato esclusivamente su immagini di bottiglie sane. In questo modo impara a rappresentare la struttura normale dell'oggetto.

Durante il test, le regioni che si discostano dalla normalità appresa possono essere ricostruite con maggiore difficoltà e produrre un errore più elevato.

---

## 🛠️ Architettura: CNN-Autoencoder

Il cuore del progetto è un **Autoencoder convoluzionale** formato da due componenti principali:

* **Encoder**: comprime l'immagine originale in una rappresentazione latente di dimensione `128 × 8 × 8`, conservando le caratteristiche principali delle bottiglie sane.
* **Decoder**: utilizza la rappresentazione compressa per ricostruire l'immagine originale pixel per pixel.

L'encoder utilizza strati `Conv2d`, `BatchNorm2d` e `LeakyReLU`.

Il decoder utilizza strati `ConvTranspose2d`, `BatchNorm2d`, `ReLU` e una funzione `Sigmoid` finale per produrre valori compresi tra 0 e 1.

Quando viene analizzata un'immagine difettosa, la differenza tra immagine originale e ricostruita viene utilizzata per individuare le regioni potenzialmente anomale.

---

## 📊 Pipeline di lavoro

1. **Dataset**: utilizzo del dataset MVTec Anomaly Detection.
2. **Categoria selezionata**: `bottle`.
3. **Preprocessing**: ridimensionamento delle immagini a `128 × 128` pixel e conversione in tensori PyTorch.
4. **Suddivisione dei dati sani**:

   * 167 immagini per il training;
   * 42 immagini per la validation.
5. **Training**: addestramento effettuato esclusivamente sulle immagini `good`, minimizzando la Mean Squared Error.
6. **Selezione del modello**: vengono mantenuti i pesi associati alla validation loss minima.
7. **Errore di ricostruzione**: calcolo della differenza assoluta tra immagine originale e ricostruita.
8. **Heatmap**: media dell'errore sui tre canali RGB e applicazione di un filtro `GaussianBlur`.
9. **Regione di interesse**: applicazione di una maschera circolare per ridurre gli artefatti sul bordo esterno.
10. **Anomaly score**: calcolo del 99° percentile dei valori della heatmap.
11. **Soglia**: calcolata come 95° percentile degli anomaly score delle immagini sane di validation.
12. **Maschera binaria**: selezione dei pixel con errore superiore alla soglia.
13. **Post-processing**: apertura e chiusura morfologica con kernel `3 × 3`.
14. **Valutazione finale**: calcolo di metriche image-level e pixel-level sul test set.

---

## 🗂️ Struttura del dataset

La cartella `bottle` deve essere organizzata nel seguente modo:

```text
bottle/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── broken_large/
│   ├── broken_small/
│   └── contamination/
└── ground_truth/
    ├── broken_large/
    ├── broken_small/
    └── contamination/
```

Le immagini presenti in `ground_truth` vengono utilizzate esclusivamente per valutare la qualità della localizzazione delle anomalie.

---

## 📈 Risultati ottenuti

Il sistema genera una visualizzazione composta da:

* **Immagine originale**;
* **Immagine ricostruita**;
* **Heatmap dell'errore**;
* **Maschera binaria sovrapposta all'immagine**;
* **Ground truth**, quando disponibile.

### Performance di training

* Epoche: `100`
* Batch size: `16`
* Ottimizzatore: `Adam`
* Learning rate: `0.001`
* Loss: `MSELoss`
* Training loss finale: `0.000279`
* Validation loss minima: `0.000293`
* Soglia finale: `0.0490`

### Metriche image-level

* Accuracy: `0.8313`
* Precision: `0.9623`
* Recall: `0.8095`
* F1-score: `0.8793`
* Specificità: `0.9000`

### Metriche pixel-level

* IoU media: `0.2104`
* Precision media: `0.5956`
* Recall media: `0.2515`

### Risultati per categoria

* `broken_large`: 20 anomalie rilevate su 20;
* `broken_small`: 19 anomalie rilevate su 22;
* `contamination`: 12 anomalie rilevate su 21.

Il sistema risulta particolarmente efficace nel riconoscimento delle rotture strutturali, mentre le contaminazioni rappresentano la categoria più complessa.

---

## 🚀 Requisiti e installazione

Per eseguire il progetto è necessario Python 3 e le seguenti librerie:

```bash
pip install torch torchvision matplotlib numpy opencv-python
```

---

## ▶️ Esecuzione

Posizionare la cartella `bottle` nella stessa directory dello script ed eseguire:

```bash
python autoencoder.py
```

Il programma addestra automaticamente il modello, calibra la soglia, valuta il test set e salva i risultati.

---

## 📂 Output generati

I file vengono salvati nella cartella:

```text
output_finale/
```

La cartella contiene:

* curva di training e validation loss;
* visualizzazioni delle immagini analizzate;
* heatmap dell'errore;
* maschere delle anomalie;
* confronto con le ground truth;
* modello addestrato `autoencoder_bottle.pth`.

---

## 🧩 Struttura del codice

* **Preparazione dei dati**: caricamento e suddivisione delle immagini sane.
* **CNN-Autoencoder**: definizione di encoder, bottleneck e decoder.
* **Training**: ottimizzazione tramite MSELoss e Adam.
* **Validation**: selezione dei pesi con validation loss minima.
* **Calibrazione della soglia**: utilizzo esclusivo delle immagini sane di validation.
* **Post-processing**: Gaussian Blur, ROI circolare e operazioni morfologiche.
* **Inferenza**: classificazione delle immagini e generazione delle maschere.
* **Metriche**: accuracy, precision, recall, F1-score, specificità, IoU e metriche pixel-level.
* **Visualizzazione**: salvataggio degli output nella cartella `output_finale`.
