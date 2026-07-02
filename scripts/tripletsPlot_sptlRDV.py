#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERADOR DE GRÁFICA: TOP 20 TRIPLETES (AP)
Modelo: SpatioTemporalRDV
"""

import os, sys, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import ivtmetrics
import warnings
warnings.filterwarnings("ignore")

root_dir = r"C:\ProyectoRDV"
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from src.tcn import CirugiaTCN

# ── Configuración ─────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(root_dir, "data", "CholecT50")
FEATURES_DIR = os.path.join(root_dir, "data", "features_1d_custom")
CAMS_DIR     = os.path.join(root_dir, "data", "cams_offline")
CKPT_TCN     = os.path.join(root_dir, "checkpoints", "TCN_FINAL.pth")
CKPT_FUSION  = os.path.join(root_dir, "checkpoints", "SpatioTemporal_FINAL.pth")

TEST_VIDEOS = ["VID79", "VID02", "VID51", "VID06", "VID25", 
               "VID14", "VID66", "VID23", "VID50", "VID111"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Arquitectura Local ────────────────────────────────────────────────────────
class SpatialRefinementModule(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
        )
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, cam, tcn_logits):
        tcn_w = torch.sigmoid(tcn_logits).unsqueeze(-1).unsqueeze(-1)
        cam_m = cam * tcn_w
        x     = self.fusion_conv(torch.cat([cam_m, cam], dim=1))
        return x + self.residual_weight * cam

class FusionHeadTrainer(nn.Module):
    def __init__(self):
        super().__init__()
        self.refine_inst = SpatialRefinementModule(num_classes=6)
        self.refine_verb = SpatialRefinementModule(num_classes=10)
        self.refine_targ = SpatialRefinementModule(num_classes=15)
        self.triplet_head = nn.Sequential(
            nn.Conv2d(31, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 100, kernel_size=1, bias=False),
        )

    def forward(self, cam_i, cam_v, cam_t, tcn_i, tcn_v, tcn_t):
        r_i   = self.refine_inst(cam_i, tcn_i)
        r_v   = self.refine_verb(cam_v, tcn_v)
        r_t   = self.refine_targ(cam_t, tcn_t)
        r_ivt = self.triplet_head(torch.cat([r_i, r_v, r_t], dim=1))
        return r_ivt

# ── Funciones Auxiliares ──────────────────────────────────────────────────────
gap = nn.AdaptiveAvgPool2d((1, 1))
sigmoid = nn.Sigmoid()

def cam_to_prob(cam: torch.Tensor) -> torch.Tensor:
    return sigmoid(gap(cam).squeeze(-1).squeeze(-1))

def norm(x):
    return (x - x.mean()) / (x.std() + 1e-6)

# ── Cargar Modelos e ivtmetrics ───────────────────────────────────────────────
print("-> Cargando Arquitectura Espacio-Temporal...")
modelo = FusionHeadTrainer().to(device)
ckpt   = torch.load(CKPT_FUSION, map_location=device, weights_only=False)
modelo.load_state_dict(ckpt["fusion_head_state"])
modelo.eval()

tcn = CirugiaTCN().to(device)
tcn_ckpt = torch.load(CKPT_TCN, map_location=device, weights_only=False)
tcn.load_state_dict(tcn_ckpt.get("model_state_dict", tcn_ckpt))
tcn.eval()

rec_ivt = ivtmetrics.Recognition(100)

# ── Extracción de Probabilidades ──────────────────────────────────────────────
print("\n-> Evaluando métricas del Triplete. Por favor espere...")

with torch.no_grad():
    for nombre in TEST_VIDEOS:
        cams_i = norm(np.load(os.path.join(CAMS_DIR, f"cams_inst_{nombre}.npy")).astype(np.float32))
        cams_v = norm(np.load(os.path.join(CAMS_DIR, f"cams_verb_{nombre}.npy")).astype(np.float32))
        cams_t = norm(np.load(os.path.join(CAMS_DIR, f"cams_targ_{nombre}.npy")).astype(np.float32))
        
        feats = np.load(os.path.join(FEATURES_DIR, f"{nombre}.npy")).astype(np.float32)
        ft    = torch.from_numpy(feats).T.unsqueeze(0).to(device)
        out_i, out_v, out_t, _ = tcn(ft)
        
        pesos_i, pesos_v, pesos_t = out_i.squeeze(0).T, out_v.squeeze(0).T, out_t.squeeze(0).T
        triplet_labels = np.loadtxt(os.path.join(DATA_DIR, "triplet", f"{nombre}.txt"), dtype=int, delimiter=",")[:, 1:]
        
        T = min(cams_i.shape[0], pesos_t.shape[0], len(triplet_labels))
        
        BATCH = 64
        for start in range(0, T, BATCH):
            end = min(start + BATCH, T)
            
            ci, cv, ct = torch.from_numpy(cams_i[start:end]).to(device), torch.from_numpy(cams_v[start:end]).to(device), torch.from_numpy(cams_t[start:end]).to(device)
            ti, tv, tt = pesos_i[start:end], pesos_v[start:end], pesos_t[start:end]
            
            r_ivt = modelo(ci, cv, ct, ti, tv, tt)
            prob_ivt = cam_to_prob(r_ivt).cpu()
            
            y_ivt = torch.from_numpy(triplet_labels[start:end].astype(np.float32))
            rec_ivt.update(y_ivt, prob_ivt)
            
        rec_ivt.video_end()

# ── Extraer APs y Ordenar ─────────────────────────────────────────────────────
resultados_ivt = rec_ivt.compute_video_AP(ignore_null=False)
mAP_global = resultados_ivt['mAP'] * 100

# Extraer el AP por clase (100 clases) y convertir a porcentaje
aps_por_clase = np.array(resultados_ivt['AP']) * 100

# Limpiar las clases que devolvieron NaN (porque no aparecen en los videos de prueba)
aps_por_clase[np.isnan(aps_por_clase)] = 0.0

# Obtener los índices de los 20 mejores tripletes
top_20_idx = np.argsort(aps_por_clase)[-20:]
top_20_aps = aps_por_clase[top_20_idx]
top_20_labels = [f"Triplete_{i}" for i in top_20_idx]

# ── Generar Gráfica de Barras (Estilo Tesis) ──────────────────────────────────
print("-> Generando figura del Top 20...")
plt.figure(figsize=(14, 10))

# Colores degradados en azul
colores = sns.color_palette("Blues_r", len(top_20_aps))

# Dibujar barras horizontales
y_pos = np.arange(len(top_20_aps))
barras = plt.barh(y_pos, top_20_aps, color=colores, edgecolor='black', alpha=0.9)

# Etiquetas y formato
plt.yticks(y_pos, top_20_labels, fontsize=11)
plt.xlabel("Precisión Promedio (AP) %", fontsize=12, fontweight='bold')
plt.ylabel("Clase de Triplete <Instrumento, Verbo, Órgano>", fontsize=12, fontweight='bold')
plt.title(f"Rendimiento Espaciotemporal: Top 20 Tripletes Quirúrgicos (mAP Global: {mAP_global:.2f}%)", 
          fontsize=16, fontweight='bold', pad=20)

# Poner el texto del porcentaje al lado de cada barra
for barra in barras:
    ancho = barra.get_width()
    plt.text(ancho + 0.5, barra.get_y() + barra.get_height()/2, f"{ancho:.1f}%", 
             va='center', ha='left', fontsize=11, fontweight='bold')

plt.xlim(0, 100)
plt.grid(axis='x', linestyle='--', alpha=0.6)

# Eliminar bordes superior y derecho para mayor limpieza visual
sns.despine()

plt.tight_layout()
plt.show()

print("[OK] ¡Proceso terminado!")