'''import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2

# =========================
# CONFIG 
# =========================
DATASET_PATH = "./bottle" 
IMAGE_SIZE = 128
BATCH_SIZE = 16 # Ridotto per stabilità con BatchNorm
EPOCHS = 50     # Aumentate per una ricostruzione più nitida
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# TRASFORMAZIONI 
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Caricamento selettivo: Solo "good" per il training 
train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "train"), transform=transform)
train_dataset.samples = [s for s in train_dataset.samples if "good" in s[0]]
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# =========================
# AUTOENCODER MIGLIORATO 
# =========================
class ImprovedAutoencoder(nn.Module):
    def __init__(self):
        super(ImprovedAutoencoder, self).__init__()
        # Encoder con BatchNorm per dettagli più definiti
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), # 8x8 (Spazio Latente) 
            nn.LeakyReLU(0.2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), # 64x64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), # 128x128
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

model = ImprovedAutoencoder().to(DEVICE)
criterion = nn.MSELoss() # Minimizzazione errore ricostruzione 
optimizer = optim.Adam(model.parameters(), lr=LR)

# =========================
# TRAINING 
# =========================
print("Inizio addestramento su soli oggetti sani...")
for epoch in range(EPOCHS):
    model.train()
    loss_val = 0
    for imgs, _ in train_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_val += loss.item()
    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss_val/len(train_loader):.6f}")

# =========================
# INFERENCE & HEATMAP 
# =========================
model.eval()
with torch.no_grad():
    for idx, (img, label) in enumerate(test_loader):
        img = img.to(DEVICE)
        recon = model(img)
        
        # Conversione per visualizzazione
        img_np = img.cpu().squeeze().permute(1,2,0).numpy()
        recon_np = recon.cpu().squeeze().permute(1,2,0).numpy()
        
        # 1. Calcolo Errore (Heatmap grezza) 
        diff = np.abs(img_np - recon_np)
        heatmap = np.mean(diff, axis=2) # Media dei canali colore
        
        # 2. Post-processing per evidenziare il difetto 
        heatmap_blur = cv2.GaussianBlur(heatmap, (5,5), 0)
        # Normalizzazione locale
        heatmap_norm = (heatmap_blur - heatmap_blur.min()) / (heatmap_blur.max() - heatmap_blur.min() + 1e-8)
        
        # Visualizzazione finale 
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img_np)
        axes[0].set_title("Originale (Test)")
        axes[1].imshow(recon_np)
        axes[1].set_title("Ricostruito (Sano)")
        # Usiamo 'jet' per la heatmap classica (rosso = errore alto)
        axes[2].imshow(heatmap_norm, cmap='jet')
        axes[2].set_title("Mappa dell'Anomalia")
        
        for ax in axes: ax.axis('off')
        
        plt.savefig(os.path.join(OUTPUT_DIR, f"result_{idx}.png"))
        if idx < 3: plt.show() # Mostra solo i primi 3 a schermo
        else: plt.close()
        
        if idx >= 10: break # Limite per test rapido'''




import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2

# =========================
# 1. CONFIGURAZIONE
# =========================
DATASET_PATH = "./bottle"  # Percorso della cartella 'bottle' del dataset MVTec AD
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50 
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "./output_finale"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# 2. PREPARAZIONE DATI
# =========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

# Training: Carichiamo solo le immagini sane ('good')
train_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "train"), transform=transform)
train_dataset.samples = [s for s in train_dataset.samples if "good" in s[0]]
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Test: Carichiamo tutto (sani e difettosi) per la valutazione
test_dataset = datasets.ImageFolder(root=os.path.join(DATASET_PATH, "test"), transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

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
# 4. TRAINING LOOP
# =========================
print(f"Inizio addestramento su {len(train_dataset)} immagini sane...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
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
print("\nCalibrazione soglia sulle immagini di test sane...")

with torch.no_grad():
    for img, label in test_loader:
        # Usiamo le immagini 'good' del test set per definire il limite della normalità
        if test_dataset.classes[label] == "good":
            img_dev = img.to(DEVICE)
            recon = model(img_dev)
            diff = np.abs(img.squeeze().permute(1,2,0).numpy() - recon.cpu().squeeze().permute(1,2,0).numpy())
            heatmap = np.mean(diff, axis=2)
            heatmap_blur = cv2.GaussianBlur(heatmap, (5,5), 0)
            error_values.append(np.max(heatmap_blur))

# La soglia viene impostata al 95° percentile degli errori sui sani per evitare falsi positivi
threshold = np.percentile(error_values, 95)
print(f"Soglia calcolata per lo scoring: {threshold:.4f}")

# =========================
# 6. INFERENZA E VISUALIZZAZIONE FINALE
# =========================
print("\nGenerazione risultati e mappatura anomalie...")
with torch.no_grad():
    for idx, (img, label) in enumerate(test_loader):
        img_device = img.to(DEVICE)
        recon = model(img_device)
        
        img_np = img.squeeze().permute(1,2,0).numpy()
        recon_np = recon.cpu().squeeze().permute(1,2,0).numpy()
        
        # Calcolo Errore Pixel-per-Pixel
        diff = np.abs(img_np - recon_np)
        heatmap = np.mean(diff, axis=2)
        heatmap_blur = cv2.GaussianBlur(heatmap, (5,5), 0)
        
        # Maschera Binaria (Scoring basato sulla soglia)
        binary_mask = (heatmap_blur > threshold).astype(np.float32)
        
        # Visualizzazione a 4 pannelli (come da obiettivo finale)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(img_np)
        axes[0].set_title("Originale (Test)")
        
        axes[1].imshow(recon_np)
        axes[1].set_title("Ricostruito (Sano)")
        
        # Heatmap a colori (Jet) per l'operatore umano
        h_min, h_max = heatmap_blur.min(), heatmap_blur.max()
        heatmap_norm = (heatmap_blur - h_min) / (h_max - h_min + 1e-8)
        axes[2].imshow(heatmap_norm, cmap='jet')
        axes[2].set_title("Mappa Errore (Heatmap)")
        
        # Decisione binaria dell'IA
        axes[3].imshow(binary_mask, cmap='gray')
        axes[3].set_title(f"Anomalia Rilevata\n(Soglia > {threshold:.3f})")
        
        for ax in axes: ax.axis('off')
        
        plt.savefig(os.path.join(OUTPUT_DIR, f"result_{idx}.png"))
        if idx < 3: plt.show()
        else: plt.close()
        
        if idx >= 20: break 

print(f"Processo completato. Immagini salvate in: {OUTPUT_DIR}")