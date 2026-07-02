#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GENERADOR DE MATRICES DE CONFUSIÓN (MULTI-LABEL)
Especialista: Acciones Quirúrgicas (Verbos)
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

print("Iniciando Auditoría Visual: Generación de Matrices de Confusión para Verbos...")

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
data_dir        = r"C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\CholecT50"
dataset_variant = 'cholect45-crossval'
version         = 100

# AQUÍ VA LA RUTA DE TU MODELO DE VERBOS (El que logró el 54% en la Época 14)
# Asegúrate de ajustar la ruta exacta a tu entorno
ckpt_path       = r"C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\CholecT50\rendezvous\pytorch\__checkpoint__\run_verbos_100\Rendezvous_ESPECIALISTA_VERBOS.pth"
imagen_salida   = "matrices_confusion_verbos.png"

# Las 10 acciones del dataset CholecT50
verb_names = ['Grasp', 'Retract', 'Dissect', 'Coagulate', 'Clip', 'Cut', 'Aspirate', 'Irrigate', 'Pack', 'Null']

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
print("Despertando al Especialista en Verbos (CAGAM)...")
model = network.Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()

# Cargar los pesos del Especialista en Acciones
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

print("Realizando examen visual a las acciones... (Esto tomará unos minutos)")
with torch.no_grad():
    # Nota: y2 contiene las etiquetas de los verbos
    for batch, (img, (_, y2, _, _)) in enumerate(val_dataloader):
        img = img.cuda()
        
        # Obtener predicciones
        _, verb, _, _ = model(img)
        _, logit_v = verb
        
        # Convertir a probabilidades (0.0 a 1.0)
        prob_v = activation(logit_v).cpu().numpy()
        
        # Convertir a binario (1 si la probabilidad es mayor a 50%, sino 0)
        preds_binarias = (prob_v > 0.5).astype(int)
        targets_reales = y2.numpy().astype(int)
        
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

# Lienzo panorámico de 2 filas y 5 columnas
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
axes = axes.ravel()

# Paleta verde quirófano para diferenciarlo de los instrumentos
cmap = sns.color_palette("Greens", as_cmap=True)

for i, (nombre_verbo, matriz) in enumerate(zip(verb_names, matrices)):
    sns.heatmap(matriz, annot=True, fmt='d', cmap=cmap, ax=axes[i], 
                cbar=False, annot_kws={"size": 14},
                xticklabels=['Ausente', 'Presente'], 
                yticklabels=['Ausente', 'Presente'])
    
    axes[i].set_title(f'{nombre_verbo}', fontsize=16, fontweight='bold')
    axes[i].set_xlabel('Predicción de la IA', fontsize=12)
    axes[i].set_ylabel('Realidad (Médico)', fontsize=12)

plt.tight_layout()
plt.savefig(imagen_salida, dpi=300, bbox_inches='tight')
print(f"\n¡Éxito Total! La gráfica de acciones se ha guardado como: {imagen_salida}")