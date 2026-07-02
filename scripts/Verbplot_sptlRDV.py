#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GENERADOR DE MATRICES DE CONFUSIÓN - VERBOS (ACCIONES)
Modelo: SpatioTemporalRDV
"""

import os, sys, torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
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

# Videos del Test Fold 1 Oficial
TEST_VIDEOS = ["VID79", "VID02", "VID51", "VID06", "VID25", 
               "VID14", "VID66", "VID23", "VID50", "VID111"]

NOMBRES_VERB = ['Grasp', 'Retract', 'Dissect', 'Coagulate', 'Clip', 
                'Cut', 'Aspirate', 'Irrigate', 'Pack', 'Null']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Arquitectura Local (Optimizada para Inferencia Rápida) ────────────────────
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
        return r_i, r_v, r_t

# ── Funciones Auxiliares ──────────────────────────────────────────────────────
gap = nn.AdaptiveAvgPool2d((1, 1))
sigmoid = nn.Sigmoid()

def cam_to_prob(cam: torch.Tensor) -> torch.Tensor:
    return sigmoid(gap(cam).squeeze(-1).squeeze(-1))

def norm(x):
    return (x - x.mean()) / (x.std() + 1e-6)

# ── Cargar Modelos ────────────────────────────────────────────────────────────
print("-> Cargando Arquitectura...")
modelo = FusionHeadTrainer().to(device)
ckpt   = torch.load(CKPT_FUSION, map_location=device, weights_only=False)
modelo.load_state_dict(ckpt["fusion_head_state"])
modelo.eval()

tcn = CirugiaTCN().to(device)
tcn_ckpt = torch.load(CKPT_TCN, map_location=device, weights_only=False)
tcn.load_state_dict(tcn_ckpt.get("model_state_dict", tcn_ckpt))
tcn.eval()

# ── Acumuladores de Métricas ──────────────────────────────────────────────────
# 10 clases para los verbos
metricas = {i: {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} for i in range(10)}

print("\n-> Procesando videos para Matrices de Confusión (Verbos)...")

with torch.no_grad():
    for nombre in TEST_VIDEOS:
        # Cargar características
        cams_i = norm(np.load(os.path.join(CAMS_DIR, f"cams_inst_{nombre}.npy")).astype(np.float32))
        cams_v = norm(np.load(os.path.join(CAMS_DIR, f"cams_verb_{nombre}.npy")).astype(np.float32))
        cams_t = norm(np.load(os.path.join(CAMS_DIR, f"cams_targ_{nombre}.npy")).astype(np.float32))
        
        feats = np.load(os.path.join(FEATURES_DIR, f"{nombre}.npy")).astype(np.float32)
        ft    = torch.from_numpy(feats).T.unsqueeze(0).to(device)
        out_i, out_v, out_t, _ = tcn(ft)
        
        pesos_i, pesos_v, pesos_t = out_i.squeeze(0).T, out_v.squeeze(0).T, out_t.squeeze(0).T
        
        # Etiquetas Reales de VERBOS
        verb_labels = np.loadtxt(os.path.join(DATA_DIR, "verb", f"{nombre}.txt"), dtype=int, delimiter=",")[:, 1:]
        
        T = min(cams_i.shape[0], pesos_v.shape[0], len(verb_labels))
        
        # Inferencia
        ci, cv, ct = torch.from_numpy(cams_i[:T]).to(device), torch.from_numpy(cams_v[:T]).to(device), torch.from_numpy(cams_t[:T]).to(device)
        _, r_v, _  = modelo(ci, cv, ct, pesos_i[:T], pesos_v[:T], pesos_t[:T])
        
        # Binarizar predicciones (Umbral 0.5)
        probs_v = cam_to_prob(r_v).cpu().numpy()
        preds_v = (probs_v >= 0.5).astype(int)
        y_true  = verb_labels[:T]
        
        # Acumular resultados por clase (10 clases)
        for c in range(10):
            metricas[c]['TP'] += np.sum((preds_v[:, c] == 1) & (y_true[:, c] == 1))
            metricas[c]['TN'] += np.sum((preds_v[:, c] == 0) & (y_true[:, c] == 0))
            metricas[c]['FP'] += np.sum((preds_v[:, c] == 1) & (y_true[:, c] == 0))
            metricas[c]['FN'] += np.sum((preds_v[:, c] == 0) & (y_true[:, c] == 1))

# ── Generar Gráficos (Estilo Tesis) ───────────────────────────────────────────
print("-> Generando figura...")
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.subplots_adjust(hspace=0.4, wspace=0.4)

for idx, ax in enumerate(axes.flatten()):
    cm = np.array([
        [metricas[idx]['TN'], metricas[idx]['FP']],
        [metricas[idx]['FN'], metricas[idx]['TP']]
    ])
    
    # Usamos un tono verde ('Greens') para diferenciarlo de los instrumentos (azules)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=ax, annot_kws={"size": 12})
    
    ax.set_title(NOMBRES_VERB[idx], fontsize=13, fontweight='bold')
    ax.set_xticklabels(['Ausente', 'Presente'], fontsize=9)
    ax.set_yticklabels(['Ausente', 'Presente'], fontsize=9)
    
    if idx >= 5: # Solo poner label X en la fila de abajo
        ax.set_xlabel('Predicción de la IA', fontsize=10)
    if idx % 5 == 0: # Solo poner label Y en la primera columna
        ax.set_ylabel('Realidad (Médico)', fontsize=10)

plt.suptitle('Matrices de Confusión de Acciones (Verbos) - SpatioTemporalRDV', fontsize=18, fontweight='bold', y=0.98)
plt.show()

print("[OK] ¡Proceso terminado!")