import copy
import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# ============================================================
# 0. RIPRODUCIBILITÀ
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Rende più riproducibili le esecuzioni su GPU.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ============================================================
# 1. CONFIGURAZIONE
# ============================================================

DATASET_PATH = "./bottle"
OUTPUT_DIR = "./output_finale"

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3

# Regione di interesse circolare.
ROI_RADIUS_RATIO = 0.45

# Lo score di un'immagine è il 99° percentile della heatmap.
IMAGE_SCORE_PERCENTILE = 99.0

# La soglia finale è il 95° percentile degli score sani.
THRESHOLD_PERCENTILE = 95.0

# Numero di figure mostrate durante l'esecuzione.
# Tutte le figure vengono comunque salvate.
SHOW_FIRST_N = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Device: {DEVICE}")


# ============================================================
# 2. PREPARAZIONE DEI DATI
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


# ------------------------------------------------------------
# Immagini sane disponibili nella cartella train/good
# ------------------------------------------------------------

full_good_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "train"),
    transform=transform
)

# Mantiene esplicitamente soltanto la classe good.
good_samples = [
    sample
    for sample in full_good_dataset.samples
    if os.path.basename(os.path.dirname(sample[0])) == "good"
]

if not good_samples:
    raise RuntimeError(
        "Nessuna immagine trovata nella cartella train/good."
    )

# Aggiornamento coerente degli attributi di ImageFolder.
full_good_dataset.samples = good_samples
full_good_dataset.imgs = good_samples
full_good_dataset.targets = [
    label for _, label in good_samples
]


# ------------------------------------------------------------
# Suddivisione 80% training e 20% validation
# ------------------------------------------------------------

train_size = int(0.80 * len(full_good_dataset))
val_size = len(full_good_dataset) - train_size

split_generator = torch.Generator().manual_seed(SEED)

train_dataset, val_dataset = random_split(
    full_good_dataset,
    [train_size, val_size],
    generator=split_generator
)


# ------------------------------------------------------------
# DataLoader
# ------------------------------------------------------------

loader_generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=loader_generator,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


# ------------------------------------------------------------
# Test set completamente separato
# ------------------------------------------------------------

test_dataset = datasets.ImageFolder(
    root=os.path.join(DATASET_PATH, "test"),
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)


print(f"\nImmagini sane totali : {len(full_good_dataset)}")
print(f"Training             : {len(train_dataset)}")
print(f"Validation           : {len(val_dataset)}")
print(f"Test                 : {len(test_dataset)}")


# ============================================================
# 3. ARCHITETTURA CNN-AUTOENCODER
# ============================================================

class AnomalyDetector(nn.Module):
    """
    CNN-Autoencoder addestrato esclusivamente su immagini sane.

    Input:
        3 x 128 x 128

    Spazio latente:
        128 x 8 x 8

    Output:
        3 x 128 x 128
    """

    def __init__(self):
        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # 128 -> 64 -> 32 -> 16 -> 8
        # ----------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                32,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                64,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(
                128,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )

        # ----------------------------------------------------
        # Decoder
        # 8 -> 16 -> 32 -> 64 -> 128
        # ----------------------------------------------------

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                128,
                128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(
                128,
                64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(
                64,
                32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.ConvTranspose2d(
                32,
                3,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        latent_representation = self.encoder(x)
        reconstruction = self.decoder(latent_representation)
        return reconstruction


model = AnomalyDetector().to(DEVICE)

criterion = nn.MSELoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 4. FUNZIONI UTILI
# ============================================================

def tensor_to_numpy_image(tensor):
    """
    Converte un tensore PyTorch 1 x C x H x W
    in un array NumPy H x W x C.
    """

    return (
        tensor
        .detach()
        .cpu()
        .squeeze(0)
        .permute(1, 2, 0)
        .numpy()
    )


def create_circle_mask(shape, radius_ratio=0.45):
    """
    Crea una regione di interesse circolare centrata.

    I pixel esterni alla ROI vengono esclusi dalla heatmap
    per ridurre gli errori periferici dovuti al contrasto
    tra la bottiglia scura e lo sfondo bianco.
    """

    height, width = shape

    mask = np.zeros(
        (height, width),
        dtype=np.float32
    )

    center = (
        width // 2,
        height // 2
    )

    radius = int(
        min(height, width) * radius_ratio
    )

    cv2.circle(
        mask,
        center,
        radius,
        1.0,
        thickness=-1
    )

    return mask


def compute_heatmap(
    original_image,
    reconstructed_image,
    roi_mask
):
    """
    Calcola la heatmap dell'errore di ricostruzione:

    1. differenza assoluta sui canali RGB;
    2. media sui tre canali;
    3. Gaussian Blur 5 x 5;
    4. applicazione della ROI circolare.
    """

    difference = np.abs(
        original_image - reconstructed_image
    )

    heatmap = np.mean(
        difference,
        axis=2
    )

    heatmap = cv2.GaussianBlur(
        heatmap,
        (5, 5),
        0
    )

    heatmap = heatmap * roi_mask

    return heatmap


def compute_anomaly_score(
    heatmap,
    percentile=99.0
):
    """
    Restituisce il percentile alto della heatmap.

    Il 99° percentile è meno sensibile del massimo
    a un singolo pixel rumoroso, ma conserva sensibilità
    verso anomalie relativamente piccole.
    """

    return float(
        np.percentile(
            heatmap,
            percentile
        )
    )


def clean_binary_mask(binary_mask):
    """
    Pulisce la maschera mediante:

    1. apertura morfologica;
    2. chiusura morfologica.

    Viene utilizzato un kernel quadrato 3 x 3.
    """

    binary_uint8 = (
        binary_mask * 255
    ).astype(np.uint8)

    kernel = np.ones(
        (3, 3),
        dtype=np.uint8
    )

    cleaned_mask = cv2.morphologyEx(
        binary_uint8,
        cv2.MORPH_OPEN,
        kernel
    )

    cleaned_mask = cv2.morphologyEx(
        cleaned_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return (
        cleaned_mask > 0
    ).astype(np.float32)


def compute_metrics(pred_mask, gt_mask):
    """
    Calcola IoU, precision e recall pixel-level.
    """

    prediction = pred_mask.astype(bool)
    ground_truth = gt_mask.astype(bool)

    intersection = np.logical_and(
        prediction,
        ground_truth
    ).sum()

    union = np.logical_or(
        prediction,
        ground_truth
    ).sum()

    iou = intersection / (
        union + 1e-8
    )

    precision = intersection / (
        prediction.sum() + 1e-8
    )

    recall = intersection / (
        ground_truth.sum() + 1e-8
    )

    return iou, precision, recall


def get_ground_truth_path(image_path):
    """
    Converte, per esempio:

    bottle/test/broken_large/001.png

    in:

    bottle/ground_truth/broken_large/001_mask.png
    """

    ground_truth_path = image_path.replace(
        os.sep + "test" + os.sep,
        os.sep + "ground_truth" + os.sep
    )

    base_path, _ = os.path.splitext(
        ground_truth_path
    )

    return base_path + "_mask.png"


def load_ground_truth(ground_truth_path):
    """
    Carica una ground truth e la ridimensiona con
    interpolazione nearest-neighbor.

    Questa interpolazione preserva la natura binaria
    della maschera.
    """

    ground_truth = cv2.imread(
        ground_truth_path,
        cv2.IMREAD_GRAYSCALE
    )

    if ground_truth is None:
        raise RuntimeError(
            "Impossibile leggere la ground truth: "
            f"{ground_truth_path}"
        )

    ground_truth = cv2.resize(
        ground_truth,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_NEAREST
    )

    return (
        ground_truth > 0
    ).astype(np.float32)


# ============================================================
# 5. ADDESTRAMENTO
# ============================================================

print(
    f"\nInizio addestramento per {EPOCHS} epoche..."
)

train_losses = []
val_losses = []

best_val_loss = float("inf")
best_model_state = None


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()
    total_train_loss = 0.0

    for images, _ in train_loader:
        images = images.to(DEVICE)

        reconstructions = model(images)

        loss = criterion(
            reconstructions,
            images
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    average_train_loss = (
        total_train_loss / len(train_loader)
    )


    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    model.eval()
    total_val_loss = 0.0

    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(DEVICE)

            reconstructions = model(images)

            validation_loss = criterion(
                reconstructions,
                images
            )

            total_val_loss += validation_loss.item()

    average_val_loss = (
        total_val_loss / len(val_loader)
    )

    train_losses.append(
        average_train_loss
    )

    val_losses.append(
        average_val_loss
    )


    # --------------------------------------------------------
    # Salvataggio in memoria dei pesi migliori
    # --------------------------------------------------------

    if average_val_loss < best_val_loss:
        best_val_loss = average_val_loss

        best_model_state = copy.deepcopy(
            model.state_dict()
        )


    if epoch == 0 or (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1:3d}/{EPOCHS}] | "
            f"Train Loss: {average_train_loss:.6f} | "
            f"Val Loss: {average_val_loss:.6f}"
        )


if best_model_state is None:
    raise RuntimeError(
        "Non è stato possibile memorizzare i pesi del modello."
    )


# Caricamento dei pesi con validation loss più bassa.
model.load_state_dict(
    best_model_state
)

print(
    "Pesi con validation loss minima caricati "
    f"(Val Loss={best_val_loss:.6f})."
)


# ------------------------------------------------------------
# Grafico delle curve di convergenza
# ------------------------------------------------------------

loss_figure = plt.figure(
    figsize=(8, 4)
)

plt.plot(
    range(1, len(train_losses) + 1),
    train_losses,
    label="Train Loss"
)

plt.plot(
    range(1, len(val_losses) + 1),
    val_losses,
    label="Val Loss"
)

plt.xlabel("Epoca")
plt.ylabel("MSE Loss")
plt.title("Curve di convergenza")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "loss_curves.png"
    ),
    dpi=150,
    bbox_inches="tight"
)

plt.close(loss_figure)


# ============================================================
# 6. CALIBRAZIONE DELLA SOGLIA
# ============================================================

model.eval()

circle_mask = create_circle_mask(
    (IMAGE_SIZE, IMAGE_SIZE),
    radius_ratio=ROI_RADIUS_RATIO
)

validation_scores = []

print(
    "\nCalibrazione soglia sul validation set sano..."
)


with torch.no_grad():
    for image, _ in val_loader:

        reconstruction = model(
            image.to(DEVICE)
        )

        image_np = tensor_to_numpy_image(
            image
        )

        reconstruction_np = tensor_to_numpy_image(
            reconstruction
        )

        heatmap = compute_heatmap(
            image_np,
            reconstruction_np,
            circle_mask
        )

        anomaly_score = compute_anomaly_score(
            heatmap,
            IMAGE_SCORE_PERCENTILE
        )

        validation_scores.append(
            anomaly_score
        )


if not validation_scores:
    raise RuntimeError(
        "Validation set vuoto: impossibile calcolare la soglia."
    )


threshold = float(
    np.percentile(
        validation_scores,
        THRESHOLD_PERCENTILE
    )
)


print(
    f"Immagini usate per calibrazione : "
    f"{len(validation_scores)}"
)

print(
    f"Score min / medio / max (val)   : "
    f"{min(validation_scores):.4f} / "
    f"{np.mean(validation_scores):.4f} / "
    f"{max(validation_scores):.4f}"
)

print(
    f"Soglia ({THRESHOLD_PERCENTILE:.0f}° percentile)       : "
    f"{threshold:.4f}"
)


# ============================================================
# 7. INFERENZA, METRICHE E VISUALIZZAZIONE
# ============================================================

print("\nInferenza sul test set...")


true_positives = 0
true_negatives = 0
false_positives = 0
false_negatives = 0

ious = []
pixel_precisions = []
pixel_recalls = []

predicted_anomalies = 0


category_statistics = {
    class_name: {
        "total": 0,
        "detected": 0
    }
    for class_name in test_dataset.classes
    if class_name != "good"
}


with torch.no_grad():
    for index, (image, label) in enumerate(test_loader):

        label_name = test_dataset.classes[
            label.item()
        ]

        image_path, _ = test_dataset.samples[
            index
        ]

        reconstruction = model(
            image.to(DEVICE)
        )

        image_np = tensor_to_numpy_image(
            image
        )

        reconstruction_np = tensor_to_numpy_image(
            reconstruction
        )


        # ----------------------------------------------------
        # Heatmap e anomaly score
        # ----------------------------------------------------

        heatmap = compute_heatmap(
            image_np,
            reconstruction_np,
            circle_mask
        )

        anomaly_score = compute_anomaly_score(
            heatmap,
            IMAGE_SCORE_PERCENTILE
        )


        true_is_anomaly = (
            label_name != "good"
        )

        predicted_is_anomaly = (
            anomaly_score > threshold
        )

        predicted_label = (
            "ANOMALA"
            if predicted_is_anomaly
            else "NORMALE"
        )

        if predicted_is_anomaly:
            predicted_anomalies += 1


        # ----------------------------------------------------
        # Matrice di confusione image-level
        # ----------------------------------------------------

        if true_is_anomaly and predicted_is_anomaly:
            true_positives += 1

        elif (
            not true_is_anomaly
            and not predicted_is_anomaly
        ):
            true_negatives += 1

        elif (
            not true_is_anomaly
            and predicted_is_anomaly
        ):
            false_positives += 1

        else:
            false_negatives += 1


        # ----------------------------------------------------
        # Statistiche image-level per categoria
        # ----------------------------------------------------

        if true_is_anomaly:
            category_statistics[
                label_name
            ]["total"] += 1

            if predicted_is_anomaly:
                category_statistics[
                    label_name
                ]["detected"] += 1


        # ----------------------------------------------------
        # Maschera binaria pixel-level
        # ----------------------------------------------------

        binary_mask = (
            heatmap > threshold
        ).astype(np.float32)

        binary_mask = clean_binary_mask(
            binary_mask
        )


        # ----------------------------------------------------
        # Ground truth e metriche pixel-level
        # ----------------------------------------------------

        ground_truth_mask = None

        if true_is_anomaly:

            ground_truth_path = get_ground_truth_path(
                image_path
            )

            if os.path.exists(ground_truth_path):

                ground_truth_mask = load_ground_truth(
                    ground_truth_path
                )

                (
                    image_iou,
                    image_pixel_precision,
                    image_pixel_recall
                ) = compute_metrics(
                    binary_mask,
                    ground_truth_mask
                )

                ious.append(
                    image_iou
                )

                pixel_precisions.append(
                    image_pixel_precision
                )

                pixel_recalls.append(
                    image_pixel_recall
                )

                print(
                    f"[{index:3d}] "
                    f"{label_name:20s} | "
                    f"Score={anomaly_score:.4f} | "
                    f"IoU={image_iou:.3f} | "
                    f"P={image_pixel_precision:.3f} | "
                    f"R={image_pixel_recall:.3f}"
                )

            else:
                print(
                    f"[{index:3d}] "
                    f"{label_name:20s} | "
                    f"Score={anomaly_score:.4f} | "
                    f"GT non trovata: {ground_truth_path}"
                )

        else:
            print(
                f"[{index:3d}] "
                f"{'good':20s} | "
                f"Score={anomaly_score:.4f} | "
                f"→ {predicted_label}"
            )


        # ----------------------------------------------------
        # Normalizzazione della heatmap solo per visualizzazione
        # ----------------------------------------------------

        heatmap_min = float(
            heatmap.min()
        )

        heatmap_max = float(
            heatmap.max()
        )

        normalized_heatmap = (
            heatmap - heatmap_min
        ) / (
            heatmap_max
            - heatmap_min
            + 1e-8
        )


        # ----------------------------------------------------
        # Visualizzazione
        # ----------------------------------------------------

        number_of_panels = (
            5 if ground_truth_mask is not None else 4
        )

        figure, axes = plt.subplots(
            1,
            number_of_panels,
            figsize=(5 * number_of_panels, 5)
        )


        axes[0].imshow(
            np.clip(image_np, 0, 1)
        )

        axes[0].set_title(
            "Originale (Test)"
        )


        axes[1].imshow(
            np.clip(reconstruction_np, 0, 1)
        )

        axes[1].set_title(
            "Ricostruito"
        )


        axes[2].imshow(
            normalized_heatmap,
            cmap="jet",
            vmin=0,
            vmax=1
        )

        axes[2].set_title(
            "Heatmap errore"
        )


        axes[3].imshow(
            np.clip(image_np, 0, 1)
        )

        # I pixel normali vengono resi trasparenti.
        visible_mask = np.ma.masked_where(
            binary_mask == 0,
            binary_mask
        )

        axes[3].imshow(
            visible_mask,
            cmap="jet",
            alpha=0.55,
            vmin=0,
            vmax=1
        )

        axes[3].set_title(
            f"Predizione: {predicted_label}\n"
            f"Score={anomaly_score:.4f} | "
            f"Thr={threshold:.4f}"
        )


        if ground_truth_mask is not None:

            axes[4].imshow(
                ground_truth_mask,
                cmap="gray",
                vmin=0,
                vmax=1
            )

            axes[4].set_title(
                "Ground Truth"
            )


        for axis in axes:
            axis.axis("off")


        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"result_{index:03d}.png"
            ),
            dpi=150,
            bbox_inches="tight"
        )


        if index < SHOW_FIRST_N:
            plt.show()

        plt.close(figure)


# ============================================================
# 8. RIEPILOGO FINALE
# ============================================================

total_images = (
    true_positives
    + true_negatives
    + false_positives
    + false_negatives
)


accuracy = (
    true_positives + true_negatives
) / (
    total_images + 1e-8
)


image_precision = true_positives / (
    true_positives
    + false_positives
    + 1e-8
)


image_recall = true_positives / (
    true_positives
    + false_negatives
    + 1e-8
)


image_f1 = (
    2
    * image_precision
    * image_recall
    / (
        image_precision
        + image_recall
        + 1e-8
    )
)


specificity = true_negatives / (
    true_negatives
    + false_positives
    + 1e-8
)


print("\n" + "=" * 55)
print("RISULTATI FINALI")
print("=" * 55)

print(f"Soglia usata         : {threshold:.4f}")
print(f"Immagini analizzate  : {total_images}")
print(f"Predette anomale     : {predicted_anomalies}")


print("\n--- Metriche Image-Level ---")

print(
    f"TP={true_positives}  "
    f"TN={true_negatives}  "
    f"FP={false_positives}  "
    f"FN={false_negatives}"
)

print(f"Accuracy   : {accuracy:.4f}")
print(f"Precision  : {image_precision:.4f}")
print(f"Recall     : {image_recall:.4f}")
print(f"F1-score   : {image_f1:.4f}")
print(f"Specificità: {specificity:.4f}")


if ious:

    print(
        "\n--- Metriche Pixel-Level "
        "(immagini anomale con GT) ---"
    )

    print(
        f"IoU media      : "
        f"{np.mean(ious):.4f}"
    )

    print(
        f"Precision media: "
        f"{np.mean(pixel_precisions):.4f}"
    )

    print(
        f"Recall media   : "
        f"{np.mean(pixel_recalls):.4f}"
    )

else:
    print(
        "\nNessuna metrica pixel-level calcolata."
    )


print("\n--- Risultati per categoria ---")

for category_name, statistics in category_statistics.items():

    total = statistics["total"]
    detected = statistics["detected"]

    category_recall = detected / (
        total + 1e-8
    )

    print(
        f"{category_name:20s} | "
        f"Rilevate={detected}/{total} | "
        f"Recall={category_recall:.4f}"
    )


# ============================================================
# 9. SALVATAGGIO DEL MODELLO
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "autoencoder_bottle.pth"
)


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "threshold": threshold,
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "roi_radius_ratio": ROI_RADIUS_RATIO,
        "image_score_percentile": IMAGE_SCORE_PERCENTILE,
        "threshold_percentile": THRESHOLD_PERCENTILE,
        "best_validation_loss": best_val_loss,
        "seed": SEED
    },
    model_path
)


print(f"\nOutput salvato in: {OUTPUT_DIR}")
print(f"Modello salvato in: {model_path}")
