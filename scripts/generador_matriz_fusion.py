#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EVALUADOR DE FUSIÓN Y GENERADOR DE MÉTRICAS (TRIPLETES)
Fase Final: Rendezvous End-to-End
"""

import os
import torch
import network
import dataloader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings("ignore") # Para ignorar advertencias de clases con 0 ejemplos

print("Iniciando Evaluación de Fusión: Inferencia de Tripletes Quirúrgicos...")

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS Y HARDWARE
# ==========================================
data_dir        = r"C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\CholecT50"
dataset_variant = 'cholect45-crossval'
version         = 100

# RUTA AL CHECKPOINT DE FUSIÓN (El modelo con el mejor AP_IVT)
ckpt_path       = r"C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\CholecT50\rendezvous\pytorch\__checkpoint__\run_fusion_100\Rendezvous_FUSION_FINAL.pth"
imagen_salida   = "rendimiento_tripletes_top20.png"

# ==========================================
# 2. CARGAR DATASET DE VALIDACIÓN
# ==========================================
print("Cargando videos de validación...")
dataset = dataloader.CholecT50( 
            dataset_dir=data_dir, 
            dataset_variant=dataset_variant,
            test_fold=1, 
            augmentation_list=['original']
            )
_, val_dataset, _ = dataset.build()

# DataLoader optimizado para arquitecturas móviles/gaming
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

# Extraer el diccionario de nombres de los tripletes si está disponible en tu dataloader
# (Si tu dataloader no lo exporta directo, usaremos índices numéricos)
try:
    triplet_dict = dataset.triplet_names 
except:
    triplet_dict = {i: f"Triplete_{i}" for i in range(100)}

# ==========================================
# 3. PREPARAR LA RED NEURONAL (FUSIÓN)
# ==========================================
print("Despertando al Nodo de Rendezvous (Fusión Total)...")
model = network.Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()

if os.path.exists(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado. Récord IVT: {checkpoint.get('benchmark', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
else:
    print(f"ERROR: No se encontró el archivo del modelo en {ckpt_path}")
    exit()

model.eval()
activation = torch.nn.Sigmoid()

# ==========================================
# 4. BUCLE DE INFERENCIA (TRIPLETES)
# ==========================================
all_preds_ivt = []
all_targets_ivt = []

print("Calculando probabilidades condicionales del triplete... (Por favor espere)")
with torch.no_grad():
    # y4 contiene las etiquetas de los 100 tripletes
    for batch, (img, (_, _, _, y4)) in enumerate(val_dataloader):
        img = img.cuda()
        
        # El cuarto elemento de la salida de RDV corresponde a los tripletes
        _, _, _, ivt = model(img)
        
        # EXTRACCIÓN SEGURA DE LOGITS
        if isinstance(ivt, tuple):
            logit_ivt = ivt[-1] # Toma el último elemento de la tupla (los logits)
        else:
            logit_ivt = ivt     # Si no es tupla, es directamente el tensor
        
        # Convertir a probabilidades (0.0 a 1.0)
        prob_ivt = activation(logit_ivt).cpu().numpy()
        targets_reales = y4.numpy().astype(int)
        
        all_preds_ivt.append(prob_ivt)
        all_targets_ivt.append(targets_reales)
        
        if batch % 50 == 0:
            print(f"   Analizando lote {batch}/{len(val_dataloader)}...")

all_preds_ivt = np.vstack(all_preds_ivt)
all_targets_ivt = np.vstack(all_targets_ivt)

# ==========================================
# 5. CÁLCULO DE AP (AVERAGE PRECISION)
# ==========================================
print("\nCalculando Precisión Promedio (AP) por clase de Triplete...")
ap_por_triplete = []
clases_validas = []

for i in range(100): # 100 clases de tripletes en CholecT50
    # Solo calcular si la clase realmente aparece en el set de validación
    if np.sum(all_targets_ivt[:, i]) > 0:
        ap = average_precision_score(all_targets_ivt[:, i], all_preds_ivt[:, i])
        ap_por_triplete.append((triplet_dict.get(i, f"Triplete {i}"), ap))
        clases_validas.append(ap)

# Calcular el mAP global
map_ivt_global = np.mean(clases_validas)
print(f"\n========================================")
print(f"🏅 mAP GLOBAL DEL TRIPLETE (IVT): {map_ivt_global:.4f}")
print(f"========================================\n")

# ==========================================
# 6. GRAFICACIÓN ACADÉMICA (TOP 20)
# ==========================================
# Ordenar de mayor a menor rendimiento
ap_por_triplete.sort(key=lambda x: x[1], reverse=True)
top_20 = ap_por_triplete[:20]

nombres = [x[0] for x in top_20]
valores_ap = [x[1] * 100 for x in top_20] # Convertir a porcentaje

# Crear la figura (Paleta Azul/Púrpura para la fase de Fusión)
plt.figure(figsize=(14, 10))
sns.set_style("whitegrid")
paleta = sns.color_palette("Blues_r", len(valores_ap))

grafica = sns.barplot(x=valores_ap, y=nombres, palette=paleta, edgecolor=".2")

# Decoración de la gráfica
plt.title(f'Rendimiento de Fusión: Top 20 Tripletes Quirúrgicos (mAP Global: {map_ivt_global*100:.2f}%)', 
          fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Precisión Promedio (AP) %', fontsize=14, fontweight='bold')
plt.ylabel('Clase de Triplete <Instrumento, Verbo, Órgano>', fontsize=14, fontweight='bold')
plt.xlim(0, 100)

# Añadir las etiquetas de porcentaje al final de cada barra
for p in grafica.patches:
    width = p.get_width()
    plt.text(width + 1.5, p.get_y() + p.get_height()/2. + 0.2,
             f'{width:.1f}%', ha="center", va="center", fontsize=11, fontweight='bold', color='black')

plt.tight_layout()
plt.savefig(imagen_salida, dpi=300, bbox_inches='tight')
print(f"¡Auditoría Finalizada! La gráfica se ha guardado como: {imagen_salida}")