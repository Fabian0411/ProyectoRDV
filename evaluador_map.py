#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUADOR OFICIAL DE PRECISION PROMEDIO (mAP)
Pipeline offline: CAMs pre-calculadas + pesos TCN pre-calculados
                  + FusionHeadTrainer entrenado.

Evaluacion sobre el Split de Prueba oficial (cholect50-crossval, fold 1):
Videos de test: VID02, VID06, VID14, VID23, VID25, VID50, VID51, VID66, VID79, VID111

Metrica: ivtmetrics v0.1.5 — compute_video_AP con ignore_null=False
"""

import os, sys, torch, time
import numpy as np
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")

root_dir = r"C:\ProyectoRDV"
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from src.tcn import CirugiaTCN
import ivtmetrics

# ── Configuracion ─────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(root_dir, "data", "CholecT50")
FEATURES_DIR = os.path.join(root_dir, "data", "features_1d_custom")
CAMS_DIR     = os.path.join(root_dir, "data", "cams_offline")
CKPT_TCN     = os.path.join(root_dir, "checkpoints", "TCN_FINAL.pth")
CKPT_FUSION  = os.path.join(root_dir, "checkpoints", "SpatioTemporal_FINAL.pth")

DATASET_VARIANT = "cholect50-crossval"
TEST_FOLD       = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── SpatialRefinementModule CORREGIDO (sin relu_final) ───────────────────────
class SpatialRefinementModule(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
        )
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, cam, tcn_logits):
        tcn_w = torch.sigmoid(tcn_logits).unsqueeze(-1).unsqueeze(-1)
        cam_m = cam * tcn_w
        x     = self.fusion_conv(torch.cat([cam_m, cam], dim=1))
        return x + self.residual_weight * cam   # sin ReLU final


# ── FusionHeadTrainer (identico al usado en entrenamiento) ────────────────────
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
        return r_i, r_v, r_t, r_ivt


# ── Splits oficiales (mismo orden que dataloader.py) ─────────────────────────
SPLITS = {
    "cholect50-crossval": {
        1: [79,  2, 51,  6, 25, 14, 66, 23, 50, 111],
        2: [80, 32,  5, 15, 40, 47, 26, 48, 70,  96],
        3: [31, 57, 36, 18, 52, 68, 10,  8, 73, 103],
        4: [42, 29, 60, 27, 65, 75, 22, 49, 12, 110],
        5: [78, 43, 62, 35, 74,  1, 56,  4, 13,  92],
    },
    "cholect50": {
        "test": [6, 51, 10, 73, 14, 74, 32, 80, 42, 111]
    },
}

test_ids   = SPLITS[DATASET_VARIANT][TEST_FOLD]
test_videos = ["VID{}".format(str(v).zfill(2)) for v in test_ids]


# ── Global Average Pool (una sola instancia) ──────────────────────────────────
gap      = nn.AdaptiveAvgPool2d((1, 1))
sigmoid  = nn.Sigmoid()


def cam_to_prob(cam: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, C]  con sigmoid aplicado"""
    return sigmoid(gap(cam).squeeze(-1).squeeze(-1))


# ── Cargar FusionHeadTrainer ──────────────────────────────────────────────────
print(f"-> Dispositivo: {device}")
print("-> Cargando FusionHeadTrainer...")

modelo = FusionHeadTrainer().to(device)
ckpt   = torch.load(CKPT_FUSION, map_location=device, weights_only=False)

if "fusion_head_state" not in ckpt:
    print("[X] ERROR: el checkpoint no contiene 'fusion_head_state'.")
    print(f"    Claves disponibles: {list(ckpt.keys())}")
    sys.exit(1)

modelo.load_state_dict(ckpt["fusion_head_state"])
modelo.eval()
print("   [+] Pesos de fusión cargados.")

# ── Cargar TCN para calcular pesos temporales ─────────────────────────────────
print("-> Cargando TCN...")
tcn      = CirugiaTCN().to(device)
tcn_ckpt = torch.load(CKPT_TCN, map_location=device, weights_only=False)
tcn.load_state_dict(tcn_ckpt.get("model_state_dict", tcn_ckpt))
tcn.eval()
print("   [+] TCN lista.")

# ── Inicializar métricas oficiales ────────────────────────────────────────────
print("-> Inicializando ivtmetrics...")
rec_i   = ivtmetrics.Recognition(6)
rec_v   = ivtmetrics.Recognition(10)
rec_t   = ivtmetrics.Recognition(15)
rec_ivt = ivtmetrics.Recognition(100)

# ── Verificar videos disponibles ─────────────────────────────────────────────
print(f"\n-> Test fold {TEST_FOLD} ({DATASET_VARIANT})")
print(f"   Videos esperados : {test_videos}")

videos_ok      = []
videos_faltante = []
for v in test_videos:
    tiene_cams = all(os.path.exists(os.path.join(CAMS_DIR, f"cams_{r}_{v}.npy"))
                     for r in ["inst", "verb", "targ"])
    tiene_feat = os.path.exists(os.path.join(FEATURES_DIR, f"{v}.npy"))
    tiene_txt  = all(os.path.exists(os.path.join(DATA_DIR, cat, f"{v}.txt"))
                     for cat in ["triplet", "instrument", "verb", "target"])
    if tiene_cams and tiene_feat and tiene_txt:
        videos_ok.append(v)
    else:
        videos_faltante.append(v)
        falta = []
        if not tiene_cams: falta.append("CAMs")
        if not tiene_feat: falta.append("features")
        if not tiene_txt:  falta.append("labels .txt")
        print(f"   [!] {v}: faltan {', '.join(falta)}")

if not videos_ok:
    print("\n[X] No hay videos evaluables. Verifica las rutas.")
    sys.exit(1)

print(f"   Videos evaluables: {len(videos_ok)}/{len(test_videos)}")
if videos_faltante:
    print(f"   Omitidos         : {videos_faltante}")

# ── Bucle de evaluacion ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EVALUACION OFICIAL — SPATIOTEMPORALRDV")
print("=" * 55)

t_total = time.time()

with torch.no_grad():
    for nombre in videos_ok:
        t_vid = time.time()
        print(f"\n  >> {nombre}...")

        # ── Cargar CAMs y normalizar (igual que en entrenamiento) ─────────────
        cams_i = np.load(os.path.join(CAMS_DIR,
                          f"cams_inst_{nombre}.npy")).astype(np.float32)
        cams_v = np.load(os.path.join(CAMS_DIR,
                          f"cams_verb_{nombre}.npy")).astype(np.float32)
        cams_t = np.load(os.path.join(CAMS_DIR,
                          f"cams_targ_{nombre}.npy")).astype(np.float32)

        # Normalizar por video — identico al DatasetCAMs del entrenamiento
        def norm(x):
            return (x - x.mean()) / (x.std() + 1e-6)

        cams_i = norm(cams_i)   # [T, 6,  7, 7]
        cams_v = norm(cams_v)   # [T, 10, 7, 7]
        cams_t = norm(cams_t)   # [T, 15, 7, 7]

        # ── Calcular pesos TCN para toda la secuencia ─────────────────────────
        feats = np.load(os.path.join(FEATURES_DIR,
                         f"{nombre}.npy")).astype(np.float32)   # [T, 512]
        ft    = torch.from_numpy(feats).T.unsqueeze(0).to(device)  # [1,512,T]
        out_i, out_v, out_t, _ = tcn(ft)
        # [1, C, T] → [T, C]
        pesos_i = out_i.squeeze(0).T   # [T, 6]
        pesos_v = out_v.squeeze(0).T   # [T, 10]
        pesos_t = out_t.squeeze(0).T   # [T, 15]

        T = min(cams_i.shape[0], pesos_i.shape[0])

        # ── Cargar etiquetas desde .txt (formato oficial del benchmark) ───────
        triplet_labels = np.loadtxt(
            os.path.join(DATA_DIR, "triplet",    f"{nombre}.txt"),
            dtype=int, delimiter=","
        )[:, 1:]   # quitar columna de índice de frame

        tool_labels = np.loadtxt(
            os.path.join(DATA_DIR, "instrument", f"{nombre}.txt"),
            dtype=int, delimiter=","
        )[:, 1:]

        verb_labels = np.loadtxt(
            os.path.join(DATA_DIR, "verb",       f"{nombre}.txt"),
            dtype=int, delimiter=","
        )[:, 1:]

        target_labels = np.loadtxt(
            os.path.join(DATA_DIR, "target",     f"{nombre}.txt"),
            dtype=int, delimiter=","
        )[:, 1:]

        T = min(T, len(triplet_labels))

        # ── Inferencia frame a frame en batches ───────────────────────────────
        BATCH = 64

        for start in range(0, T, BATCH):
            end  = min(start + BATCH, T)
            bs   = end - start

            ci = torch.from_numpy(cams_i[start:end]).to(device)   # [B, 6,  7,7]
            cv = torch.from_numpy(cams_v[start:end]).to(device)   # [B, 10, 7,7]
            ct = torch.from_numpy(cams_t[start:end]).to(device)   # [B, 15, 7,7]
            ti = pesos_i[start:end]                                # [B, 6]
            tv = pesos_v[start:end]                                # [B, 10]
            tt = pesos_t[start:end]                                # [B, 15]

            r_i, r_v, r_t, r_ivt = modelo(ci, cv, ct, ti, tv, tt)

            p_i   = cam_to_prob(r_i).cpu()    # [B, 6]
            p_v   = cam_to_prob(r_v).cpu()    # [B, 10]
            p_t   = cam_to_prob(r_t).cpu()    # [B, 15]
            p_ivt = cam_to_prob(r_ivt).cpu()  # [B, 100]

            y_i   = torch.from_numpy(tool_labels[start:end].astype(np.float32))
            y_v   = torch.from_numpy(verb_labels[start:end].astype(np.float32))
            y_t   = torch.from_numpy(target_labels[start:end].astype(np.float32))
            y_ivt = torch.from_numpy(triplet_labels[start:end].astype(np.float32))

            rec_i.update(y_i,   p_i)
            rec_v.update(y_v,   p_v)
            rec_t.update(y_t,   p_t)
            rec_ivt.update(y_ivt, p_ivt)

        rec_i.video_end()
        rec_v.video_end()
        rec_t.video_end()
        rec_ivt.video_end()

        t_v = time.time() - t_vid
        print(f"     {T} frames procesados en {t_v:.1f}s")

# ── Resultados finales ────────────────────────────────────────────────────────
ap_i   = rec_i.compute_video_AP(ignore_null=False)["mAP"]   * 100
ap_v   = rec_v.compute_video_AP(ignore_null=False)["mAP"]   * 100
ap_t   = rec_t.compute_video_AP(ignore_null=False)["mAP"]   * 100
ap_ivt = rec_ivt.compute_video_AP(ignore_null=False)["mAP"] * 100

t_total = time.time() - t_total

print("\n" + "=" * 55)
print("  RESULTADOS FINALES — SpatioTemporalRDV")
print(f"  Variante : {DATASET_VARIANT}, fold {TEST_FOLD}")
print(f"  Videos   : {len(videos_ok)} evaluados / {len(test_videos)} esperados")
print("=" * 55)
print(f"  AP Instrumentos (I) : {ap_i:6.2f} %")
print(f"  AP Verbos       (V) : {ap_v:6.2f} %")
print(f"  AP Organos      (T) : {ap_t:6.2f} %")
print(f"  mAP Triplete  (IVT) : {ap_ivt:6.2f} %")
print("=" * 55)
print(f"\n  Referencia — Rendezvous original [Nwoye et al. 2022]:")
print(f"  AP Instrumentos (I) :  92.00 %")
print(f"  AP Verbos       (V) :  60.70 %")
print(f"  AP Organos      (T) :  38.30 %")
print(f"  mAP Triplete  (IVT) :  29.90 %")
print("=" * 55)
print(f"\n  Tiempo total de evaluacion: {t_total:.1f}s")