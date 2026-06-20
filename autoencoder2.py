import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import cv2

# =========================
# 0. RIPRODUCIBILITÀ
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# =========================
# 1. CONFIGURAZIONE
# =========================
DATASET_PATH = "./bottle"  # Percorso della cartella 'bottle' del dataset MVTec AD
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50 
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./output_finale2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. PREPARAZIONE DATI
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Caricamento delle immagini sane disponibili in train/good
full_good_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "train"),
    transform=transform
)

# Manteniamo esplicitamente soltanto le immagini appartenenti alla classe "good"
good_samples = [
    sample
    for sample in full_good_dataset.samples
    if os.path.basename(os.path.dirname(sample[0])) == "good"
]

# Aggiornamento coerente degli attributi interni di ImageFolder
full_good_dataset.samples = good_samples
full_good_dataset.imgs = good_samples
full_good_dataset.targets = [label for _, label in good_samples]

# Divisione riproducibile: 80% training, 20% validation
train_size = int(0.80 * len(full_good_dataset))
val_size = len(full_good_dataset) - train_size

split_generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    full_good_dataset,
    [train_size, val_size],
    generator=split_generator
)

# Il training loader viene usato per ottimizzare i pesi dell'autoencoder
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

# Il validation loader viene usato esclusivamente per calibrare la soglia
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)

# Il test set resta completamente separato e viene usato solo alla fine
test_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "test"),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False
)

print("\nSuddivisione dei dati sani:")
print(f"Immagini sane totali disponibili: {len(full_good_dataset)}")
print(f"Immagini usate per il training: {len(train_dataset)}")
print(f"Immagini usate per la validation: {len(val_dataset)}")
print(f"Immagini presenti nel test set: {len(test_dataset)}")
# =========================
# 3. ARCHITETTURA CNN-AUTOENCODER
# =========================
class AnomalyDetector(nn.Module):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        # Encoder: comprime l'immagine catturando le feature essenziali del 'sano'
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),# 8x8 (Bottleneck/Spazio Latente)
            nn.LeakyReLU(0.2)
        )
        # Decoder: tenta di ricostruire l'immagine originale pixel per pixel
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # 128x128
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = AnomalyDetector().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# 4. FUNZIONI UTILI
# =========================
def create_circle_mask(shape, radius_ratio=0.42):
    """Crea una maschera circolare per ridurre l'effetto dei bordi."""
    h, w = shape
    center = (w // 2, h // 2)
    radius = int(min(h, w) * radius_ratio)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 1, -1)
    return mask
def clean_binary_mask(binary_mask):
    """Rimuove piccoli rumori con operazioni morfologiche."""
    binary_mask_uint8 = (binary_mask * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)

    binary_mask_clean = cv2.morphologyEx(binary_mask_uint8, cv2.MORPH_OPEN, kernel)
    binary_mask_clean = cv2.morphologyEx(binary_mask_clean, cv2.MORPH_CLOSE, kernel)

    return (binary_mask_clean > 0).astype(np.float32)

def compute_metrics(pred_mask, gt_mask):
    """Calcola IoU, precision, recall."""
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    iou = intersection / (union + 1e-8)
    precision = intersection / (pred.sum() + 1e-8)
    recall = intersection / (gt.sum() + 1e-8)

    return iou, precision, recall

def get_ground_truth_path(img_path):
    """
    Converte il path di un'immagine test nel path della ground truth.
    Esempio:
    bottle/test/broken_large/001.png -> bottle/ground_truth/broken_large/001_mask.png
    """
    gt_path = img_path.replace(os.sep + "test" + os.sep, os.sep + "ground_truth" + os.sep)
    base, ext = os.path.splitext(gt_path)
    gt_path = base + "_mask.png"
    return gt_path

# =========================
# 4. TRAINING LOOP
# =========================
print(f"Inizio addestramento su {len(train_dataset)} immagini sane...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.6f}")

# =========================
# 5. CALCOLO DELLA SOGLIA (THRESHOLDING)
# =========================
model.eval()
error_values = []

circle_mask = create_circle_mask(
    (IMAGE_SIZE, IMAGE_SIZE),
    radius_ratio=0.42
)

print("\nCalibrazione soglia sul validation set sano...")

with torch.no_grad():
    for img, _ in val_loader:
        img_device = img.to(DEVICE)
        recon = model(img_device)

        img_np = img.squeeze().permute(1, 2, 0).numpy()
        recon_np = recon.cpu().squeeze().permute(1, 2, 0).numpy()

        # Errore di ricostruzione pixel per pixel
        diff = np.abs(img_np - recon_np)
        heatmap = np.mean(diff, axis=2)

        # Riduzione del rumore
        heatmap_blur = cv2.GaussianBlur(
            heatmap,
            (5, 5),
            0
        )

        # Applicazione della regione di interesse
        heatmap_blur = heatmap_blur * circle_mask

        # Score image-level della singola immagine sana
        anomaly_score = np.max(heatmap_blur)
        error_values.append(anomaly_score)

if len(error_values) == 0:
    raise RuntimeError(
        "Il validation set è vuoto: impossibile calcolare la soglia."
    )

# Il 95° percentile definisce il limite superiore della normalità
threshold = np.percentile(error_values, 95)

print(f"Numero immagini usate per la calibrazione: {len(error_values)}")
print(f"Soglia calcolata sul validation set: {threshold:.4f}")

# =========================
# 6. IINFERENZA + METRICHE + VISUALIZZAZIONE
# =========================
print("\nGenerazione risultati e mappatura anomalie...")
results_summary = []
ious, precisions, recalls = [], [], []

with torch.no_grad():
    tp = tn = fp = fn = 0
    for idx, (img, label) in enumerate(test_loader):
        label_name = test_dataset.classes[label.item()]
        img_path, _ = test_dataset.samples[idx]

        img_device = img.to(DEVICE)
        recon = model(img_device)
        
        img_np = img.squeeze().permute(1,2,0).numpy()
        recon_np = recon.cpu().squeeze().permute(1,2,0).numpy()
        
        # Calcolo Errore Pixel-per-Pixel
        diff = np.abs(img_np - recon_np)
        heatmap = np.mean(diff, axis=2)
        heatmap_blur = cv2.GaussianBlur(heatmap, (5,5), 0)
        
        # Riduzione falsi positivi sul bordo
        heatmap_blur = heatmap_blur * circle_mask

        # Score numerico dell'immagine
        anomaly_score = np.max(heatmap_blur)
        image_prediction = "ANOMALA" if anomaly_score > threshold else "NORMALE"

        true_is_anomaly = (label_name != "good")
        pred_is_anomaly = (anomaly_score > threshold)

        if true_is_anomaly and pred_is_anomaly:
            tp += 1
        elif (not true_is_anomaly) and (not pred_is_anomaly):
            tn += 1
        elif (not true_is_anomaly) and pred_is_anomaly:
            fp += 1
        elif true_is_anomaly and (not pred_is_anomaly):
            fn += 1

        # Maschera Binaria (Scoring basato sulla soglia)
        binary_mask = (heatmap_blur > threshold).astype(np.float32)
        
        # Pulizia della maschera
        binary_mask = clean_binary_mask(binary_mask)

        # Salvataggio info riepilogative
        results_summary.append({
            "idx": idx,
            "path": img_path,
            "true_label": label_name,
            "score": float(anomaly_score),
            "predicted_image_label": image_prediction
        })

        # Normalizzazione heatmap solo per visualizzazione
        h_min, h_max = heatmap_blur.min(), heatmap_blur.max()
        heatmap_norm = (heatmap_blur - h_min) / (h_max - h_min + 1e-8)

        # Se l'immagine è anomala e abbiamo la ground truth, calcoliamo metriche
        gt_mask = None
        if label_name != "good":
            gt_path = get_ground_truth_path(img_path)

            if os.path.exists(gt_path):
                gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                gt_mask = cv2.resize(gt_mask, (IMAGE_SIZE, IMAGE_SIZE))
                gt_mask = (gt_mask > 0).astype(np.float32)

                iou, precision, recall = compute_metrics(binary_mask, gt_mask)
                ious.append(iou)
                precisions.append(precision)
                recalls.append(recall)

                print(f"[{idx}] {label_name} | Score={anomaly_score:.4f} | IoU={iou:.3f} | Precision={precision:.3f} | Recall={recall:.3f}")
            else:
                print(f"[{idx}] {label_name} | Score={anomaly_score:.4f} | Ground truth non trovata: {gt_path}")
        else:
            print(f"[{idx}] {label_name} | Score={anomaly_score:.4f} | Predizione immagine: {image_prediction}")


        # Visualizzazione a 4 pannelli (come da obiettivo finale)
        if gt_mask is not None:
            fig, axes = plt.subplots(1, 5, figsize=(24, 5))

            axes[0].imshow(img_np)
            axes[0].set_title("Originale (Test)")

            axes[1].imshow(recon_np)
            axes[1].set_title("Ricostruito (Sano)")

            axes[2].imshow(heatmap_norm, cmap='jet')
            axes[2].set_title("Mappa Errore (Heatmap)")

            axes[3].imshow(img_np)
            axes[3].imshow(binary_mask, cmap='jet', alpha=0.5)
            axes[3].set_title(f"{image_prediction}\nScore={anomaly_score:.3f}")

            axes[4].imshow(gt_mask, cmap='gray')
            axes[4].set_title("Ground Truth")

        else:
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))

            axes[0].imshow(img_np)
            axes[0].set_title("Originale (Test)")

            axes[1].imshow(recon_np)
            axes[1].set_title("Ricostruito (Sano)")

            axes[2].imshow(heatmap_norm, cmap='jet')
            axes[2].set_title("Mappa Errore (Heatmap)")

            axes[3].imshow(binary_mask, cmap='gray')
            axes[3].set_title(f"{image_prediction}\nScore={anomaly_score:.3f}\nThr={threshold:.3f}")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"result_{idx}.png"))

        if idx < 3:
            plt.show()
        else:
            plt.close()




# =========================
# 8. RIEPILOGO FINALE
# =========================
print("\n===== RISULTATI FINALI =====")
print(f"Soglia usata: {threshold:.4f}")
print(f"Numero immagini analizzate: {len(results_summary)}")

num_pred_anomalous = sum(1 for r in results_summary if r["predicted_image_label"] == "ANOMALA")
print(f"Numero immagini predette anomale: {num_pred_anomalous}")

if len(ious) > 0:
    print(f"IoU media: {np.mean(ious):.4f}")
    print(f"Precision media: {np.mean(precisions):.4f}")
    print(f"Recall media: {np.mean(recalls):.4f}")
else:
    print("Nessuna metrica calcolata: ground truth non disponibile nelle immagini analizzate.")

accuracy_img = (tp + tn) / (tp + tn + fp + fn + 1e-8)
precision_img = tp / (tp + fp + 1e-8)
recall_img = tp / (tp + fn + 1e-8)
f1_img = 2 * precision_img * recall_img / (precision_img + recall_img + 1e-8)

print("\n===== METRICHE IMAGE-LEVEL =====")
print(f"TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
print(f"Accuracy image-level: {accuracy_img:.4f}")
print(f"Precision image-level: {precision_img:.4f}")
print(f"Recall image-level: {recall_img:.4f}")
print(f"F1-score image-level: {f1_img:.4f}")
print(f"\nImmagini salvate in: {OUTPUT_DIR}")