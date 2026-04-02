# Anomaly Detection con Spazi Latenti (MVTec AD)

## 📌 Obiettivo del Progetto
L'obiettivo principale di questo progetto è insegnare a un'intelligenza artificiale a riconoscere **difetti di fabbrica mai visti prima**.  
Invece di addestrare il modello su esempi di errori (approccio supervisionato), utilizziamo un **approccio unsupervised** basato sulla ricostruzione dell'immagine:  X → X'


Il sistema viene addestrato **esclusivamente su oggetti sani**.  
Poiché un oggetto può essere difettoso in infiniti modi diversi, è impossibile addestrare la rete su ogni singolo errore possibile.

---

## 🛠️ Architettura: CNN-Autoencoder
Il cuore del progetto è un **Autoencoder convoluzionale** composto da due parti principali:

- **Encoder**: Comprimi l'immagine originale in un vettore compatto (spazio latente), costringendo la rete a imparare le caratteristiche essenziali di un prodotto "sano".  
- **Decoder**: Prende il vettore compresso e tenta di ricostruire l'immagine originale pixel per pixel.  

Durante il test, se inseriamo un'immagine con un difetto (es. un graffio), la rete **non saprà ricostruirlo correttamente**, producendo una versione "sana" dell'immagine.

---

## 📊 Pipeline di Lavoro

1. **Dataset**: MVTec AD, gold standard per l'Industrial Anomaly Detection.  
2. **Categoria scelta**: "bottle" per ottimizzare i tempi di training.  
3. **Training**: addestramento effettuato solo su immagini "Good" minimizzando la **Mean Squared Error (MSE) loss**.  
4. **Inference**: test su immagini con anomalie reali.  
5. **Scoring**: calcolo dell'errore di ricostruzione pixel per pixel e generazione di una **heatmap**.
6. **Soglia**: calcolata automaticamente sul 95° percentile delle immagini sane del test set per distinguere anomalie dai normali errori di ricostruzione.

---

## 📈 Risultati Ottenuti
Il modello genera una **visualizzazione affiancata** che permette di identificare immediatamente il difetto:

- **Immagine Originale**: la bottiglia con il difetto.  
- **Immagine Ricostruita**: la versione "sana" generata dall'IA.  
- **Mappa dell'Anomalia (Heatmap)**: videnzia la differenza pixel per pixel tra originale e ricostruita.
- **Maschera binaria**: segnala le anomalie rilevate in base alla soglia.

### Performance di Training
- Epoche: 50  
- Loss finale (MSE): ~0.0003  
- Tempo di training: < 1 ora (con GPU supportata)

---

## 🚀 Requisiti e Installazione
Per eseguire lo script è necessario **Python 3.x** e le seguenti librerie:

pip install torch torchvision matplotlib numpy opencv-python

---

## 📂 Struttura del Codice

- **Autoencoder**: classe definita con strati `Conv2d`, `BatchNorm2d` e `LeakyReLU` per garantire una ricostruzione nitida delle immagini.  
- **Post-processing**: utilizzo di `cv2.GaussianBlur` sulle heatmap per ridurre il rumore e isolare le anomalie strutturali.
- **Training**: loop principale con MSELoss e Adam optimizer.
- **Soglia**: calcolata automaticamente sulle immagini sane del test set.
- **Inferenza e Visualizzazione**: salvataggio delle immagini finali (Originale | Ricostruita | Heatmap | Maschera) nella cartella output_finale.