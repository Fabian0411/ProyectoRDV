#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INFERENCIA Y VISUALIZACIÓN
Dado un video y su .npy, produce frames anotados con:
- Heatmap de activación por clase (coloreado)
- Bounding boxes con nombre y confianza
"""

import os, sys, torch, cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

root_dir = r"C:\\ProyectoRDV"
sys.path.append(root_dir)

import src.network as network
from src.tcn import CirugiaTCN
from src.spatial_temporal_rdv import SpatioTemporalRDV, BoundingBoxExtractor

# ── Configuración ─────────────────────────────────────────────────────────────
NOMBRE_VIDEO = "VID01"
CKPT_RDV     = os.path.join(root_dir, "checkpoints", "run_fusion",
                             "Rendezvous_FUSION_FINAL.pth")
CKPT_TCN     = os.path.join(root_dir, "checkpoints", "TCN_FINAL.pth")
CKPT_FUSION  = os.path.join(root_dir, "checkpoints",
                             "SpatioTemporal_FINAL.pth")
FEATURES_NPY = os.path.join(root_dir, "data", "features_1d_custom",
                             f"{NOMBRE_VIDEO}.npy")
FRAMES_DIR   = os.path.join(root_dir, "data", "CholecT50", "videos",
                             NOMBRE_VIDEO)
OUTPUT_DIR   = os.path.join(root_dir, "plots", "inferencia", NOMBRE_VIDEO)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NOMBRES_INST = ['Grasper','Bipolar','Hook','Scissors','Clipper','Irrigator']
NOMBRES_VERB = ['Grasp','Retract','Dissect','Coagulate','Clip',
                'Cut','Aspirate','Irrigate','Pack','Null']
NOMBRES_TARG = ['Gallbladder','Cystic Plate','Cystic Duct','Cystic Artery',
                'Cystic Pedicle','Blood Vessel','Fluid','Adhesion','Omentum',
                'Liver','Gut','Specimen Bag','Abdominal Wall','Gauze','Null']

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# ── Cargar modelos ────────────────────────────────────────────────────────────
print("-> Cargando modelos...")
rdv = network.Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()
ckpt = torch.load(CKPT_RDV, map_location='cuda:0', weights_only=False)
rdv.load_state_dict(ckpt.get('model_state_dict', ckpt))

tcn = CirugiaTCN().cuda()
tcn_ckpt = torch.load(CKPT_TCN, map_location='cuda:0', weights_only=False)
tcn.load_state_dict(tcn_ckpt.get('model_state_dict', tcn_ckpt))

modelo = SpatioTemporalRDV(rdv, tcn).cuda()

if os.path.exists(CKPT_FUSION):
    fusion_ckpt = torch.load(CKPT_FUSION, map_location='cuda:0')
    modelo.refine_inst.load_state_dict(fusion_ckpt['refine_inst'])
    modelo.refine_verb.load_state_dict(fusion_ckpt['refine_verb'])
    modelo.refine_targ.load_state_dict(fusion_ckpt['refine_targ'])
    modelo.triplet_head.load_state_dict(fusion_ckpt['triplet_head'])
    print("   [+] Cabeza de fusión cargada.")
else:
    print("   [!] Sin checkpoint de fusión. Usando pesos iniciales.")

modelo.eval()

# ── Cargar features ───────────────────────────────────────────────────────────
features_np  = np.load(FEATURES_NPY).astype(np.float32)   # [T, 512]
features_seq = torch.from_numpy(features_np.T).unsqueeze(0).cuda()  # [1,512,T]

extractor_bb = BoundingBoxExtractor(confidence_threshold=0.3)

import glob, sorted as sorted_builtin
rutas_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.*")))

print(f"-> Procesando {len(rutas_frames)} frames de {NOMBRE_VIDEO}...")

with torch.no_grad():
    for frame_idx, ruta_frame in enumerate(rutas_frames):

        # Cargar frame original para visualización (sin normalizar)
        img_original = cv2.imread(ruta_frame)
        img_original = cv2.resize(img_original, (224, 224))

        # Preparar tensor normalizado para la red
        with Image.open(ruta_frame) as img:
            frame_tensor = TRANSFORM(img.convert('RGB')).unsqueeze(0).cuda()

        # Inferencia
        salidas = modelo(frame_tensor, features_seq, frame_idx)

        # Extraer bounding boxes de instrumentos
        bboxes = extractor_bb.extraer(
            salidas['heatmap_inst'][0],   # [6, 224, 224]
            NOMBRES_INST
        )

        # ── Construir visualización ───────────────────────────────────────────
        frame_viz = img_original.copy()

        # Superponer heatmap del instrumento más activo
        heatmap_suma = salidas['heatmap_inst'][0].sum(dim=0)  # [224,224]
        heatmap_np   = heatmap_suma.cpu().numpy()
        heatmap_np   = (heatmap_np - heatmap_np.min()) / (
                        heatmap_np.max() - heatmap_np.min() + 1e-8)
        heatmap_color = cv2.applyColorMap(
            (heatmap_np * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        frame_viz = cv2.addWeighted(frame_viz, 0.6, heatmap_color, 0.4, 0)

        # Dibujar bounding boxes
        colores = [(0,255,0),(255,0,0),(0,0,255),
                   (255,255,0),(0,255,255),(255,0,255)]
        for det in bboxes:
            x1, y1, x2, y2 = det['bbox']
            color = colores[det['clase_idx'] % len(colores)]
            cv2.rectangle(frame_viz, (x1,y1), (x2,y2), color, 2)
            etiqueta = f"{det['clase']} {det['confianza']:.2f}"
            cv2.putText(frame_viz, etiqueta, (x1, max(y1-5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Guardar frame anotado
        nombre_salida = os.path.join(
            OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg"
        )
        cv2.imwrite(nombre_salida, frame_viz)

        if (frame_idx + 1) % 100 == 0:
            print(f"   Procesados {frame_idx+1}/{len(rutas_frames)} frames")

print(f"\n[OK] Frames anotados guardados en: {OUTPUT_DIR}")
