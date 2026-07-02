#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE A: EXTRACTOR DE CARACTERÍSTICAS ESPACIALES (FEATURE MINER)
Procesa frames con el modelo Rendezvous (Época 21) y guarda
embeddings 1D [N_frames, 512] en data/features_1d_custom/.
"""

import os
import sys
import glob
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

# Enrutamiento dinámico: /scripts/../ = raíz del proyecto
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

import src.network as network

print("=========================================================")
print("   INICIANDO FASE A: MINERÍA DE VECTORES ESPACIALES      ")
print("=========================================================\n")

# ==========================================
# CONFIGURACIÓN DE RUTAS
# ==========================================
videos_dir = os.path.join(root_dir, "data", "CholecT50", "videos")
output_dir = os.path.join(root_dir, "data", "features_1d_custom")
ckpt_path  = os.path.join(root_dir, "checkpoints", "run_fusion",
                          "Rendezvous_FUSION_FINAL.pth")

os.makedirs(output_dir, exist_ok=True)

# ==========================================
# CARGA DEL MODELO
# ==========================================
print("-> Cargando modelo espacial (Época 21)...")
model = network.Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()

if not os.path.exists(ckpt_path):
    print(f"   [X] ERROR CRÍTICO: Checkpoint no encontrado en:\n       {ckpt_path}")
    print("       Verifica el nombre exacto del archivo .pth en checkpoints/run_fusion/")
    sys.exit(1)

checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
state_dict = checkpoint.get('model_state_dict', checkpoint)
model.load_state_dict(state_dict)
print("   [+] Pesos cargados con éxito.")

model.eval()

# ==========================================
# GANCHO DE EXTRACCIÓN
# ==========================================
# Captura high_level_feature antes del pooling: [1, 512, H, W]
# Pool manual → [1, 512] para alimentar la TCN
vector_actual = []
pool_global   = torch.nn.AdaptiveAvgPool2d((1, 1))

def hook_extractor(modulo, entrada, salida):
    pooled = pool_global(salida)                    # [1, 512, 1, 1]
    vector = pooled.view(pooled.size(0), -1)        # [1, 512]
    vector_actual.append(vector.detach().cpu().numpy())

try:
    capa_objetivo = model.encoder.basemodel.basemodel.layer4[1].bn2
    capa_objetivo.register_forward_hook(hook_extractor)
    print("   [+] Gancho conectado a encoder.basemodel.basemodel.layer4[1].bn2")
    print("       Shape esperado por frame: (512,)\n")
except AttributeError as e:
    print(f"   [X] Error al conectar gancho: {e}")
    sys.exit(1)

# ==========================================
# TRANSFORMACIONES (estándar ImageNet)
# ==========================================
transformacion = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==========================================
# BUCLE PRINCIPAL DE EXTRACCIÓN
# ==========================================
print("-> Iniciando escaneo de videos...")
carpetas_videos = sorted([
    d for d in os.listdir(videos_dir)
    if os.path.isdir(os.path.join(videos_dir, d))
])

if not carpetas_videos:
    print(f"   [!] No se encontraron carpetas en {videos_dir}")
    sys.exit(1)

videos_ok      = 0
videos_fallido = 0

with torch.no_grad():
    for nombre_video in carpetas_videos:
        ruta_video  = os.path.join(videos_dir, nombre_video)
        ruta_salida = os.path.join(output_dir, f"{nombre_video}.npy")

        if os.path.exists(ruta_salida):
            print(f"   [-] {nombre_video} omitido (ya existe).")
            continue

        print(f"\n   >> Procesando {nombre_video}...")

        frames = sorted(glob.glob(os.path.join(ruta_video, "*.*")))
        if not frames:
            print(f"      [!] Carpeta vacía. Saltando.")
            videos_fallido += 1
            continue

        vectores_video  = []
        frames_fallidos = 0

        for i, ruta_frame in enumerate(frames):

            # — limpiar hook antes de cada frame —
            vector_actual.clear()

            # — cargar imagen y liberar handle —
            try:
                with Image.open(ruta_frame) as img:
                    img_rgb = img.convert('RGB')
                tensor_img = transformacion(img_rgb).unsqueeze(0).cuda()
            except Exception as e:
                print(f"      [!] Frame {i} ilegible ({e}). Saltando.")
                frames_fallidos += 1
                continue

            # — forward pass —
            _ = model(tensor_img)
            del tensor_img          # liberar VRAM inmediatamente

            # — validar captura del hook —
            if len(vector_actual) == 0:
                print(f"      [!] Hook vacío en frame {i}. Saltando.")
                frames_fallidos += 1
                continue
            if len(vector_actual) > 1:
                print(f"      [!] Hook con {len(vector_actual)} capturas en frame {i}."
                      f" Usando la primera.")

            vectores_video.append(vector_actual[0][0])  # shape: (512,)

            if (i + 1) % 500 == 0:
                print(f"      Procesados {i+1}/{len(frames)} frames"
                      f"  |  fallidos: {frames_fallidos}")

        # — guardar solo si hay vectores válidos —
        if not vectores_video:
            print(f"   [X] {nombre_video}: ningún frame válido. No se guarda .npy")
            videos_fallido += 1
            continue

        matriz_final = np.vstack(vectores_video)    # [N_frames, 512]
        np.save(ruta_salida, matriz_final)

        estado = f"  |  {frames_fallidos} frames fallidos" if frames_fallidos else ""
        print(f"   [OK] {nombre_video}.npy  →  shape {matriz_final.shape}{estado}")
        videos_ok += 1

# ==========================================
# RESUMEN FINAL
# ==========================================
print("\n=========================================================")
print(f"  EXTRACCIÓN FINALIZADA")
print(f"  Videos guardados : {videos_ok}")
print(f"  Videos con error : {videos_fallido}")
print(f"  Destino          : {output_dir}")
print("=========================================================")