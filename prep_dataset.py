import os
import hashlib
import random
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuración general
# -----------------------------
SOURCE_DIR = Path("/Users/angie/OneDrive/Pictures/dataset")
TARGET_ROOT = Path("/Users/angie/OneDrive/Pictures")

SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}

MODEL_CONFIGS = {
    "resnet": {"size": (224, 224), "channels": 3},
    "densenet": {"size": (224, 224), "channels": 3},
    "efficientnet": {"size": (224, 224), "channels": 3},
    "medgemma": {"size": (320, 320), "channels": 3},
    "nuu-net": {"size": (256, 256), "channels": 1},
    "convnextlarge": {"size": (224, 224), "channels": 3},
    "vit": {"size": (224, 224), "channels": 3}
}


# -----------------------------
# Utilidades de Hash y Limpieza
# -----------------------------
def get_file_hash(path, block_size=2 ** 20):
    """Genera el hash MD5 de un archivo para detectar duplicados."""
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(block_size)
                if not b: break
                h.update(b)
        return h.hexdigest()
    except Exception as e:
        print(f"❌ Error leyendo {path}: {e}")
        return None


def get_unique_images(image_paths):
    """Filtra la lista de rutas dejando solo una instancia por cada contenido único."""
    hashes = {}
    unique_paths = []
    duplicates_count = 0

    for p in image_paths:
        file_hash = get_file_hash(p)
        if file_hash and file_hash not in hashes:
            hashes[file_hash] = p
            unique_paths.append(p)
        else:
            duplicates_count += 1

    return unique_paths, duplicates_count


# -----------------------------
# Procesamiento de Imagen
# -----------------------------
def process_and_save_image(img_path, out_path, size, channels):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return

    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if channels == 3:
        img = np.stack([img, img, img], axis=-1)

    img = img.astype(np.uint8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    Image.fromarray(img).save(out_path)


# -----------------------------
# Ejecución Principal
# -----------------------------

# 1. Escaneo inicial y limpieza de duplicados a nivel global (por clase)
print("🔍 Escaneando duplicados en el origen...")
clean_dataset = {}

for class_name in os.listdir(SOURCE_DIR):
    class_path = SOURCE_DIR / class_name
    if not class_path.is_dir(): continue

    all_images = [
        p for p in class_path.glob("*.*")
        if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
    ]

    unique_images, dup_num = get_unique_images(all_images)
    clean_dataset[class_name] = unique_images
    print(f"📂 Clase [{class_name}]: {len(unique_images)} únicas (Eliminados {dup_num} duplicados)")

# 2. Generación de datasets para cada modelo
for model_name, cfg in MODEL_CONFIGS.items():
    print(f"\n🧩 Preparando dataset para {model_name.upper()} ...")
    target_dir = TARGET_ROOT / f"dataset_{model_name}"

    for class_name, images in clean_dataset.items():
        if not images: continue

        # División de los datos únicos
        random.seed(42)
        # CORRECCIÓN: Nombres de variables consistentes
        train_val, test_imgs = train_test_split(
            images,
            test_size=SPLIT_RATIOS["test"],
            random_state=42
        )

        val_ratio_adjusted = SPLIT_RATIOS["val"] / (SPLIT_RATIOS["train"] + SPLIT_RATIOS["val"])

        train_imgs, val_imgs = train_test_split(
            train_val,
            test_size=val_ratio_adjusted,
            random_state=42
        )

        splits = {
            "train": train_imgs,
            "val": val_imgs,
            "test": test_imgs
        }

        for split_name, img_list in splits.items():
            for img_path in tqdm(img_list, desc=f"{model_name} - {class_name} ({split_name})", leave=False):
                out_path = target_dir / split_name / class_name / img_path.name
                process_and_save_image(
                    img_path,
                    out_path,
                    cfg["size"],
                    cfg["channels"]
                )

    print(f"✅ Dataset {model_name} creado en: {target_dir}")

print("\n🎉 Proceso finalizado. Todos los datasets son únicos y están listos.")