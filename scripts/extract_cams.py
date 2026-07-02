#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PASO INTERMEDIO: EXTRACTOR DE CAMs OFFLINE
Lee los frames UNA SOLA VEZ, pasa por el Rendezvous congelado
y guarda los mapas de activacion [T, C, 7, 7] como archivos .npy.

Salidas por video:
  cams_inst_VID01.npy   -> [T,  6, 7, 7]
  cams_verb_VID01.npy   -> [T, 10, 7, 7]
  cams_targ_VID01.npy   -> [T, 15, 7, 7]
"""

import os, sys, glob, torch
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

root_dir = r"C:\ProyectoRDV"
sys.path.append(root_dir)
import src.network as network

# ── Rutas ─────────────────────────────────────────────────────────────────────
VIDEOS_DIR  = os.path.join(root_dir, "data", "CholecT50", "videos")
CAMS_DIR    = os.path.join(root_dir, "data", "cams_offline")
CKPT_RDV    = os.path.join(root_dir, "checkpoints", "run_fusion",
                            "Rendezvous_FUSION_FINAL.pth")

os.makedirs(CAMS_DIR, exist_ok=True)

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Cargar Rendezvous congelado ───────────────────────────────────────────────
print("-> Cargando Rendezvous...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = network.Rendezvous('resnet18', hr_output=False, use_ln=False).to(device)
ckpt   = torch.load(CKPT_RDV, map_location=device, weights_only=False)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.eval()
print(f"   [+] Modelo listo en {device}")

# ── Bucle por video ───────────────────────────────────────────────────────────
carpetas = sorted([
    d for d in os.listdir(VIDEOS_DIR)
    if os.path.isdir(os.path.join(VIDEOS_DIR, d))
])

print(f"\n-> Procesando {len(carpetas)} videos...\n")
videos_ok = 0

with torch.no_grad():
    for nombre in carpetas:

        # Verificar si ya existe
        ruta_inst = os.path.join(CAMS_DIR, f"cams_inst_{nombre}.npy")
        if os.path.exists(ruta_inst):
            print(f"   [-] {nombre} omitido (ya existe).")
            videos_ok += 1
            continue

        frames = sorted(glob.glob(os.path.join(VIDEOS_DIR, nombre, "*.*")))
        if not frames:
            print(f"   [!] {nombre}: sin frames. Saltando.")
            continue

        print(f"   >> {nombre}  ({len(frames)} frames)...")

        cams_i_list = []
        cams_v_list = []
        cams_t_list = []

        for i, ruta_frame in enumerate(frames):
            try:
                with Image.open(ruta_frame) as img:
                    tensor = TRANSFORM(img.convert('RGB')).unsqueeze(0).to(device)
            except Exception as e:
                print(f"      [!] Frame {i} ilegible: {e}. Saltando.")
                # Insertar CAM nula para mantener alineacion temporal
                cams_i_list.append(np.zeros((6,  7, 7), dtype=np.float32))
                cams_v_list.append(np.zeros((10, 7, 7), dtype=np.float32))
                cams_t_list.append(np.zeros((15, 7, 7), dtype=np.float32))
                continue

            enc_i, enc_v, enc_t, _ = model(tensor)

            # enc_x[0] shape: [1, C, 7, 7] → quitar batch dim → [C, 7, 7]
            cams_i_list.append(enc_i[0][0].cpu().numpy())
            cams_v_list.append(enc_v[0][0].cpu().numpy())
            cams_t_list.append(enc_t[0][0].cpu().numpy())

            if (i + 1) % 500 == 0:
                print(f"      {i+1}/{len(frames)} frames procesados...")

        # Apilar: lista de [C, 7, 7] → array [T, C, 7, 7]
        np.save(os.path.join(CAMS_DIR, f"cams_inst_{nombre}.npy"),
                np.stack(cams_i_list).astype(np.float32))
        np.save(os.path.join(CAMS_DIR, f"cams_verb_{nombre}.npy"),
                np.stack(cams_v_list).astype(np.float32))
        np.save(os.path.join(CAMS_DIR, f"cams_targ_{nombre}.npy"),
                np.stack(cams_t_list).astype(np.float32))

        tam_i = os.path.getsize(
            os.path.join(CAMS_DIR, f"cams_inst_{nombre}.npy")) / 1e6

        print(f"   [OK] {nombre} guardado "
              f"| inst: {len(frames), 6, 7, 7} "
              f"| ~{tam_i:.1f} MB por rama")
        videos_ok += 1

print(f"\n{'='*50}")
print(f"  EXTRACCION FINALIZADA: {videos_ok}/{len(carpetas)} videos")
print(f"  Destino: {CAMS_DIR}")
print(f"{'='*50}")