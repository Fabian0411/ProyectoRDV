#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE B: ENTRENAMIENTO DE LA TCN (Con Validación, Early Stopping y Reanudación)
"""

import os
import sys
import glob
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from torch.utils.data import Dataset, DataLoader

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

from src.tcn import CirugiaTCN

print("=========================================================")
print("   ENTRENAMIENTO TEMPORAL (FASE B) - EARLY STOPPING      ")
print("=========================================================\n")

# ==========================================
# CONFIGURACIÓN
# ==========================================
features_dir = os.path.join(root_dir, "data", "features_1d_custom")
labels_dir   = os.path.join(root_dir, "data", "CholecT50", "labels")
ckpt_salida  = os.path.join(root_dir, "checkpoints", "TCN_FINAL.pth")
log_salida   = os.path.join(root_dir, "checkpoints", "TCN_training.log")

NUM_EPOCHS = 150
LR         = 1e-4
GRAD_CLIP  = 1.0
PATIENCE   = 10   # épocas sin mejora antes de detener

# ── Índices JSON — ajusta si tu dataset difiere ───────────────────────────────
IDX_TRIPLET = 0
IDX_INST    = 1
IDX_VERB    = 7
IDX_TARGET  = 8

# ==========================================
# SPLIT REPRODUCIBLE (80/20)
# ==========================================
todos_los_videos = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
if not todos_los_videos:
    print(f"[X] ERROR: No se encontraron .npy en {features_dir}")
    sys.exit(1)

random.Random(42).shuffle(todos_los_videos)
corte        = int(len(todos_los_videos) * 0.8)
videos_train = todos_los_videos[:corte]
videos_val   = todos_los_videos[corte:]

print(f"   Total : {len(todos_los_videos)} videos")
print(f"   Train : {len(videos_train)} | Val : {len(videos_val)}\n")

# ==========================================
# DATASET
# ==========================================
class DatasetSecuencial(Dataset):
    def __init__(self, lista_archivos, labels_dir):
        self.lista_videos = lista_archivos
        self.labels_dir   = labels_dir

    def __len__(self):
        return len(self.lista_videos)

    def __getitem__(self, idx):
        ruta_npy    = self.lista_videos[idx]
        nombre_base = os.path.splitext(os.path.basename(ruta_npy))[0]

        features   = np.load(ruta_npy).astype(np.float32)  # [T, 512]
        num_frames = features.shape[0]

        ruta_json = os.path.join(self.labels_dir, f"{nombre_base}.json")
        if not os.path.exists(ruta_json):
            raise FileNotFoundError(
                f"JSON no encontrado: {ruta_json}\n"
                f"Verifica que el archivo existe en labels_dir."
            )

        with open(ruta_json, 'r') as f:
            anotaciones = json.load(f).get("annotations", {})

        y_inst    = np.zeros((6,   num_frames), dtype=np.float32)
        y_verb    = np.zeros((10,  num_frames), dtype=np.float32)
        y_target  = np.zeros((15,  num_frames), dtype=np.float32)
        y_triplet = np.zeros((100, num_frames), dtype=np.float32)

        for frame_str, lista_acciones in anotaciones.items():
            # ── CORRECCIÓN: JSON indexa desde 1, NumPy desde 0 ───────────────
            frame_idx = int(frame_str)

            if frame_idx < 0 or frame_idx >= num_frames:
                continue

            for accion in lista_acciones:
                if len(accion) <= max(IDX_TRIPLET, IDX_INST,
                                      IDX_VERB, IDX_TARGET):
                    continue

                trip_id = accion[IDX_TRIPLET]
                inst_id = accion[IDX_INST]
                verb_id = accion[IDX_VERB]
                targ_id = accion[IDX_TARGET]

                if inst_id != -1 and inst_id < 6:
                    y_inst[inst_id, frame_idx]    = 1.0
                if verb_id != -1 and verb_id < 10:
                    y_verb[verb_id, frame_idx]    = 1.0
                if targ_id != -1 and targ_id < 15:
                    y_target[targ_id, frame_idx]  = 1.0
                if trip_id != -1 and trip_id < 100:
                    y_triplet[trip_id, frame_idx] = 1.0

        # [T, 512] → [512, T] para la TCN
        x_tensor = torch.from_numpy(features).transpose(0, 1)

        return (
            x_tensor,
            torch.from_numpy(y_inst),
            torch.from_numpy(y_verb),
            torch.from_numpy(y_target),
            torch.from_numpy(y_triplet),
            nombre_base,
        )

loader_train = DataLoader(
    DatasetSecuencial(videos_train, labels_dir),
    batch_size=1, shuffle=True,  num_workers=0
)
loader_val = DataLoader(
    DatasetSecuencial(videos_val, labels_dir),
    batch_size=1, shuffle=False, num_workers=0
)

# ==========================================
# MODELO, CRITERIO Y OPTIMIZADOR
# ==========================================
print("-> Construyendo TCN...")
model     = CirugiaTCN().cuda()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Reduce el LR a la mitad si en 5 épocas no mejora val_loss
# Esto le da una segunda oportunidad antes de que salte el early stopping
scheduler = ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5,
    patience=5
)

# ==========================================
# REANUDACIÓN DESDE CHECKPOINT
# ==========================================
epoca_inicio      = 0
mejor_loss_val    = float('inf')
epocas_sin_mejora = 0

if os.path.exists(ckpt_salida):
    print(f"\n[+] Checkpoint encontrado. Intentando reanudar...")
    ckpt = torch.load(ckpt_salida, map_location='cuda:0', weights_only=False)

    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        epoca_inicio      = ckpt['epoch']
        mejor_loss_val    = ckpt['mejor_val_loss']
        epocas_sin_mejora = ckpt.get('epocas_sin_mejora', 0)
        print(f"    Reanudando desde época {epoca_inicio + 1}")
        print(f"    Mejor val_loss hasta ahora: {mejor_loss_val:.4f}")
    else:
        # Checkpoint antiguo (solo state_dict) — carga pesos, reinicia todo lo demás
        model.load_state_dict(ckpt)
        print("    [!] Checkpoint sin metadatos (formato antiguo).")
        print("        Pesos cargados. Optimizer y scheduler reiniciados.")
        print("        Entrenamiento continuará desde época 1.")
else:
    print("[+] No se encontró checkpoint previo. Entrenamiento desde cero.")

# ==========================================
# LOG
# ==========================================
# Si reanudamos, agregamos al log existente. Si es nuevo, lo creamos.
modo_log = "a" if epoca_inicio > 0 else "w"
with open(log_salida, modo_log, encoding="utf-8") as f:
    if epoca_inicio == 0:
        f.write("Epoca,Train_Loss,Val_Loss,LR\n")

# ==========================================
# BUCLE DE ENTRENAMIENTO
# ==========================================
print(f"\n-> Entrenando desde época {epoca_inicio + 1} hasta {NUM_EPOCHS}...\n")

for epoch in range(epoca_inicio, NUM_EPOCHS):

    # ── FASE TRAIN ────────────────────────────────────────────────────────────
    model.train()
    loss_train_acum = 0.0

    for x, y_i, y_v, y_t, y_trip, nombre in loader_train:
        x      = x.cuda()
        y_i    = y_i.cuda()
        y_v    = y_v.cuda()
        y_t    = y_t.cuda()
        y_trip = y_trip.cuda()

        optimizer.zero_grad()

        out_i, out_v, out_t, out_trip = model(x)

        loss = (criterion(out_i,    y_i)
              + criterion(out_v,    y_v)
              + criterion(out_t,    y_t)
              + criterion(out_trip, y_trip) * 2.0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        loss_train_acum += loss.item()

    promedio_train = loss_train_acum / len(loader_train)

    # ── FASE VALIDACIÓN ───────────────────────────────────────────────────────
    model.eval()
    loss_val_acum = 0.0

    with torch.no_grad():
        for x, y_i, y_v, y_t, y_trip, _ in loader_val:
            x      = x.cuda()
            y_i    = y_i.cuda()
            y_v    = y_v.cuda()
            y_t    = y_t.cuda()
            y_trip = y_trip.cuda()

            out_i, out_v, out_t, out_trip = model(x)

            loss_val = (criterion(out_i,    y_i)
                      + criterion(out_v,    y_v)
                      + criterion(out_t,    y_t)
                      + criterion(out_trip, y_trip) * 2.0)

            loss_val_acum += loss_val.item()

    promedio_val = loss_val_acum / len(loader_val)
    lr_actual    = optimizer.param_groups[0]['lr']
    nuevo_record = promedio_val < mejor_loss_val

    # ── SCHEDULER ─────────────────────────────────────────────────────────────
    scheduler.step(promedio_val)

    # ── GUARDAR SI ES RÉCORD ──────────────────────────────────────────────────
    if nuevo_record:
        mejor_loss_val    = promedio_val
        epocas_sin_mejora = 0

        torch.save({
            'epoch'               : epoch + 1,
            'model_state_dict'    : model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'mejor_val_loss'      : mejor_loss_val,
            'train_loss'          : promedio_train,
            'epocas_sin_mejora'   : 0,
        }, ckpt_salida)

    else:
        epocas_sin_mejora += 1

    # ── LOG Y PRINT ───────────────────────────────────────────────────────────
    marca = " ← RÉCORD" if nuevo_record else \
            f" (paciencia: {epocas_sin_mejora}/{PATIENCE})"

    print(f"[Época {epoch+1:03d}/{NUM_EPOCHS}] "
          f"Train: {promedio_train:.4f} | "
          f"Val: {promedio_val:.4f} | "
          f"LR: {lr_actual:.2e}"
          f"{marca}")

    with open(log_salida, "a", encoding="utf-8") as f:
        f.write(f"{epoch+1},{promedio_train:.4f},"
                f"{promedio_val:.4f},{lr_actual:.2e}\n")

    # ── EARLY STOPPING ────────────────────────────────────────────────────────
    if epocas_sin_mejora >= PATIENCE:
        print(f"\n[EARLY STOPPING] Val loss sin mejorar por {PATIENCE} épocas.")
        print(f"  Mejor val_loss alcanzado : {mejor_loss_val:.4f}")
        print(f"  Checkpoint guardado en   : {ckpt_salida}")
        break

# ==========================================
# RESUMEN FINAL
# ==========================================
print("\n=========================================================")
print(f"  ENTRENAMIENTO FINALIZADO")
print(f"  Mejor val_loss : {mejor_loss_val:.4f}")
print(f"  Checkpoint     : {ckpt_salida}")
print(f"  Log            : {log_salida}")
print("=========================================================")