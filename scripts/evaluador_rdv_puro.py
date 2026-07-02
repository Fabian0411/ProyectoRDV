#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVALUADOR OFICIAL — RENDEZVOUS ESPACIAL PURO
Evalua el modelo Rendezvous entrenado (sin TCN ni cabeza de fusion)
sobre los mismos videos del test fold para comparativa directa.

Videos de test compartidos (cholect45-crossval fold 1):
  VID02, VID06, VID14, VID23, VID25, VID50, VID51, VID66, VID79

Metrica: ivtmetrics v0.1.5 — compute_video_AP con ignore_null=False
Referencia directa con evaluador_map.py (SpatioTemporalRDV).
"""

import os, sys, torch, time
import numpy as np
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")

root_dir = r"C:\ProyectoRDV"
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

import src.network as network
import ivtmetrics

# ── Configuracion ─────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(root_dir, "data", "CholecT50")
CKPT_RDV  = os.path.join(root_dir, "checkpoints", "run_fusion",
                          "Rendezvous_FUSION_FINAL.pth")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Videos de test compartidos entre ambos evaluadores ───────────────────────
# cholect45-crossval fold 1 = [79, 2, 51, 6, 25, 14, 66, 23, 50]
# cholect50-crossval fold 1 = [79, 2, 51, 6, 25, 14, 66, 23, 50, 111]
# Interseccion = los 9 videos de CholecT45, que es lo que tu RDV vio en test
TEST_IDS    = [79, 2, 51, 6, 25, 14, 66, 23, 50]
TEST_VIDEOS = ["VID{}".format(str(v).zfill(2)) for v in TEST_IDS]

# ── Transformacion identica a la usada en run_fusion.py ──────────────────────
# run_fusion usa image_width=224, image_height=128 → Resize (256,448) del dataloader
TRANSFORM = transforms.Compose([
    transforms.Resize((256, 448)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Dataset por video (igual que T50 del dataloader original) ─────────────────
from PIL import Image

class VideoDataset(torch.utils.data.Dataset):
    """
    Replica exacta de la clase T50 del dataloader original.
    Lee frames .png y etiquetas .txt del benchmark oficial.
    """
    def __init__(self, video_name, data_dir, transform):
        self.img_dir = os.path.join(data_dir, "videos", video_name)
        self.transform = transform

        self.triplet_labels = np.loadtxt(
            os.path.join(data_dir, "triplet",    f"{video_name}.txt"),
            dtype=int, delimiter=","
        )
        self.tool_labels = np.loadtxt(
            os.path.join(data_dir, "instrument", f"{video_name}.txt"),
            dtype=int, delimiter=","
        )
        self.verb_labels = np.loadtxt(
            os.path.join(data_dir, "verb",       f"{video_name}.txt"),
            dtype=int, delimiter=","
        )
        self.target_labels = np.loadtxt(
            os.path.join(data_dir, "target",     f"{video_name}.txt"),
            dtype=int, delimiter=","
        )

    def __len__(self):
        return len(self.triplet_labels)

    def __getitem__(self, idx):
        frame_id  = self.triplet_labels[idx, 0]
        basename  = "{}.png".format(str(frame_id).zfill(6))
        img_path  = os.path.join(self.img_dir, basename)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        y_i   = torch.from_numpy(self.tool_labels[idx,    1:].astype(np.float32))
        y_v   = torch.from_numpy(self.verb_labels[idx,    1:].astype(np.float32))
        y_t   = torch.from_numpy(self.target_labels[idx,  1:].astype(np.float32))
        y_ivt = torch.from_numpy(self.triplet_labels[idx, 1:].astype(np.float32))

        return image, (y_i, y_v, y_t, y_ivt)


# ── Cargar Rendezvous ─────────────────────────────────────────────────────────
print(f"-> Dispositivo: {device}")
print("-> Cargando Rendezvous espacial puro...")

model = network.Rendezvous(
    'resnet18', hr_output=False, use_ln=False
).to(device)

ckpt = torch.load(CKPT_RDV, map_location=device, weights_only=False)
model.load_state_dict(ckpt.get('model_state_dict', ckpt))
model.eval()
print("   [+] Pesos de fusion cargados.")

# ── Inicializar metricas ──────────────────────────────────────────────────────
activation = nn.Sigmoid()
rec_i   = ivtmetrics.Recognition(6)
rec_v   = ivtmetrics.Recognition(10)
rec_t   = ivtmetrics.Recognition(15)
rec_ivt = ivtmetrics.Recognition(100)

# ── Verificar videos disponibles ─────────────────────────────────────────────
print(f"\n-> Videos de test: {TEST_VIDEOS}\n")

videos_ok       = []
videos_faltante = []
for v in TEST_VIDEOS:
    tiene_imgs = os.path.isdir(os.path.join(DATA_DIR, "videos", v))
    tiene_txt  = all(os.path.exists(os.path.join(DATA_DIR, cat, f"{v}.txt"))
                     for cat in ["triplet", "instrument", "verb", "target"])
    if tiene_imgs and tiene_txt:
        videos_ok.append(v)
    else:
        videos_faltante.append(v)
        falta = []
        if not tiene_imgs: falta.append("frames")
        if not tiene_txt:  falta.append("labels .txt")
        print(f"   [!] {v}: faltan {', '.join(falta)}")

if not videos_ok:
    print("[X] No hay videos evaluables.")
    sys.exit(1)

print(f"   Videos evaluables: {len(videos_ok)}/{len(TEST_VIDEOS)}")

# ── Bucle de evaluacion ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EVALUACION OFICIAL — RENDEZVOUS ESPACIAL PURO")
print("=" * 55)

t_total = time.time()

with torch.no_grad():
    for nombre in videos_ok:
        t_vid = time.time()
        print(f"\n  >> {nombre}...")

        ds     = VideoDataset(nombre, DATA_DIR, TRANSFORM)
        loader = DataLoader(ds, batch_size=32, shuffle=False,
                            num_workers=0, pin_memory=False)

        for imgs, (y_i, y_v, y_t, y_ivt) in loader:
            imgs = imgs.to(device)

            enc_i, enc_v, enc_t, dec_ivt = model(imgs)

            # Logits del Rendezvous: enc_x[1] son los logits por componente
            prob_i   = activation(enc_i[1]).detach().cpu()
            prob_v   = activation(enc_v[1]).detach().cpu()
            prob_t   = activation(enc_t[1]).detach().cpu()
            prob_ivt = activation(dec_ivt).detach().cpu()

            rec_i.update(y_i.float(),    prob_i)
            rec_v.update(y_v.float(),    prob_v)
            rec_t.update(y_t.float(),    prob_t)
            rec_ivt.update(y_ivt.float(), prob_ivt)

        rec_i.video_end()
        rec_v.video_end()
        rec_t.video_end()
        rec_ivt.video_end()

        t_v = time.time() - t_vid
        print(f"     {len(ds)} frames procesados en {t_v:.1f}s")

# ── Resultados ────────────────────────────────────────────────────────────────
ap_i   = rec_i.compute_video_AP(ignore_null=False)["mAP"]   * 100
ap_v   = rec_v.compute_video_AP(ignore_null=False)["mAP"]   * 100
ap_t   = rec_t.compute_video_AP(ignore_null=False)["mAP"]   * 100
ap_ivt = rec_ivt.compute_video_AP(ignore_null=False)["mAP"] * 100

t_total = time.time() - t_total

print("\n" + "=" * 55)
print("  RESULTADOS — RENDEZVOUS ESPACIAL PURO")
print(f"  cholect45-crossval, fold 1 | {len(videos_ok)} videos")
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
print(f"\n  Tiempo total: {t_total:.1f}s")
print("\n  Ejecuta evaluador_map.py para la comparativa con SpatioTemporalRDV")