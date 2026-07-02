#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENERADOR DE MATRICES DE CONFUSIÓN (MULTI-LABEL)
Especialista: Órganos / Anatomía (Targets / T-CAM)
"""

import os
import torch
import network
import dataloader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import multilabel_confusion_matrix

print("Iniciando Auditoría Visual: Generación de Matrices de Confusión para Órganos...")

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
data_dir        = r"C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\CholecT50"
dataset_variant = 'cholect45-crossval'
version         = 100

# AQUÍ VA LA RUTA DE TU MODELO DE ÓRGANOS
# Asegúrate de ajustar el nombre del archivo .pth al correspondiente de tu rama anatómica
ckpt_path       = r"C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\CholecT50\rendezvous\pytorch\__checkpoint__\Copias de seguridad por si las moscas\Rendezvous_ESPECIALISTA_ORGANOS_copia.pth"
imagen_salida   = "matrices_confusion_organos.png"

# Las 15 anatomías/targets del dataset CholecT50
target_names = ['Gallbladder', 'Cystic Plate', 'Cystic Duct', 'Cystic Artery', 'Cystic Pedicle', 
                'Blood Vessel', 'Fluid', 'Wall', 'Liver', 'Adhesion', 
                'Omentum', 'Specimen', 'Gut', 'Specimen Bag', 'Null']

# ==========================================
# 2. CARGAR DATASET DE VALIDACIÓN
# ==========================================
print("Cargando videos de validación...")
dataset = dataloader.CholecT50( 
            dataset_dir=data_dir, 
            dataset_variant=dataset_variant,
            test_fold=1, 
            augmentation_list=['original'] # Sin aumentos para evaluación real
            )
_, val_dataset, _ = dataset.build()
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

# ==========================================
# 3. PREPARAR LA RED NEURONAL
# ==========================================
print("Despertando al Especialista en Órganos (T-CAM)...")
model = network.Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()

# Cargar los pesos del Especialista Anatómico
if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado desde la época {checkpoint.get('epoch', 'N/A')} con un récord de {checkpoint.get('benchmark', 'N/A'):.4f}")
    else:
        model.load_state_dict(checkpoint)
else:
    print(f"ERROR: No se encontró el archivo del modelo en {ckpt_path}")
    exit()

model.eval()
activation = torch.nn.Sigmoid()

# ==========================================
# 4. BUCLE DE INFERENCIA (EXAMEN FINAL)
# ==========================================
all_preds = []
all_targets = []

print("Realizando examen visual a la anatomía... (Esto tomará unos minutos)")
with torch.no_grad():
    # Nota: y3 contiene las etiquetas de los órganos (targets) en CholecT50
    for batch, (img, (_, _, y3, _)) in enumerate(val_dataloader):
        img = img.cuda()
        
        # Obtener predicciones (el tercer valor corresponde a los targets/órganos)
        _, _, organ, _ = model(img)
        _, logit_t = organ
        
        # Convertir a probabilidades (0.0 a 1.0)
        prob_t = activation(logit_t).cpu().numpy()
        
        # Convertir a binario (1 si la probabilidad es mayor a 50%, sino 0)
        preds_binarias = (prob_t > 0.5).astype(int)
        targets_reales = y3.numpy().astype(int)
        
        all_preds.append(preds_binarias)
        all_targets.append(targets_reales)
        
        if batch % 50 == 0:
            print(f"   Procesando lote {batch}/{len(val_dataloader)}...")

# Unir todos los lotes en matrices gigantes
all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# ==========================================
# 5. CÁLCULO Y GRAFICACIÓN
# ==========================================
print("\nCalculando Matemáticas de Confusión...")
matrices = multilabel_confusion_matrix(all_targets, all_preds)

# Lienzo panorámico de 3 filas y 5 columnas (para las 15 clases)
fig, axes = plt.subplots(3, 5, figsize=(24, 13))
axes = axes.ravel()

# Paleta cálida (Naranjas a Rojos), representativa de tejidos biológicos
cmap = sns.color_palette("OrRd", as_cmap=True)

for i, (nombre_organo, matriz) in enumerate(zip(target_names, matrices)):
    sns.heatmap(matriz, annot=True, fmt='d', cmap=cmap, ax=axes[i], 
                cbar=False, annot_kws={"size": 14},
                xticklabels=['Ausente', 'Presente'], 
                yticklabels=['Ausente', 'Presente'])
    
    axes[i].set_title(f'{nombre_organo}', fontsize=16, fontweight='bold')
    axes[i].set_xlabel('Predicción de la IA', fontsize=12)
    axes[i].set_ylabel('Realidad (Médico)', fontsize=12)

plt.tight_layout()
plt.savefig(imagen_salida, dpi=300, bbox_inches='tight')
print(f"\n¡Éxito Total! La gráfica de órganos se ha guardado como: {imagen_salida}")