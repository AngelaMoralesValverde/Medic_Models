import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Ruta de la carpeta que contiene Patient-1 ... Patient-60
input_dir = "/Users/angie/OneDrive/Pictures/dataset/archive"
output_dir = "/Users/angie/OneDrive/Pictures/dataset/ms"

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Recorrer las carpetas Patient-X (1 a 60)
for i in range(1, 61):
    patient_folder = f"Patient-{i}"
    patient_input_path = os.path.join(input_dir, patient_folder)
    patient_output_path = os.path.join(output_dir, patient_folder)

    if not os.path.isdir(patient_input_path):
        print(f"⚠ Carpeta no encontrada: {patient_input_path}")
        continue

    os.makedirs(patient_output_path, exist_ok=True)

    # Buscar archivos .nii en la carpeta del paciente
    for file in os.listdir(patient_input_path):
        if file.endswith(".nii") or file.endswith(".nii.gz"):
            nii_path = os.path.join(patient_input_path, file)
            print(f"Procesando: {nii_path}")

            # Cargar NIfTI
            nii = nib.load(nii_path)
            data = nii.get_fdata()

            # Normalización para guardar en JPG
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data = (data * 255).astype(np.uint8)

            # Guardar cada slice como imagen JPG
            num_slices = data.shape[2]
            base_name = file.replace(".nii", "").replace(".gz", "")

            for s in range(num_slices):
                slice_img = data[:, :, s]
                output_path = os.path.join(
                    patient_output_path, f"{base_name}_slice{s}.jpg"
                )
                plt.imsave(output_path, slice_img, cmap="gray")

print("✅ Conversión completada. Imágenes guardadas en: jpg_output/")
