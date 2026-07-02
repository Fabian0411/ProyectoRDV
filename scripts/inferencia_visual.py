#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
INFERENCIA Y VISUALIZACIÓN ESPACIO-TEMPORAL
Dado un video y su .npy, produce frames anotados con:
- Heatmap de activación por clase (coloreado)
- Bounding boxes con nombre y confianza de las 3 ramas.
"""

import os, sys, torch, cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

root_dir = r"C:\ProyectoRDV"
sys.path.append(root_dir)

import src.network as network
from src.tcn import CirugiaTCN
from src.spatial_temporal_rdv import SpatioTemporalRDV, BoundingBoxExtractor

# ── Configuración ─────────────────────────────────────────────────────────────
NOMBRE_VIDEO = "VID01" # Cambia esto por el video que quieras probar
CKPT_RDV     = os.path.join(root_dir, "checkpoints", "run_fusion", "Rendezvous_FUSION_FINAL.pth")
CKPT_TCN     = os.path.join(root_dir, "checkpoints", "TCN_FINAL.pth")
CKPT_FUSION  = os.path.join(root_dir, "checkpoints", "SpatioTemporal_FINAL.pth")
FEATURES_NPY = os.path.join(root_dir, "data", "features_1d_custom", f"{NOMBRE_VIDEO}.npy")
FRAMES_DIR   = os.path.join(root_dir, "data", "CholecT50", "videos", NOMBRE_VIDEO)
OUTPUT_DIR   = os.path.join(root_dir, "plots", "inferencia", NOMBRE_VIDEO)

os.makedirs(OUTPUT_DIR, exist_ok=True)

NOMBRES_INST = ['Grasper','Bipolar','Hook','Scissors','Clipper','Irrigator']
NOMBRES_VERB = ['Grasp','Retract','Dissect','Coagulate','Clip','Cut','Aspirate','Irrigate','Pack','Null']
NOMBRES_TARG = ['Gallbladder','Cystic Plate','Cystic Duct','Cystic Artery','Cystic Pedicle','Blood Vessel','Fluid','Adhesion','Omentum','Liver','Gut','Specimen Bag','Abdominal Wall','Gauze','Null']

TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Cargar modelos ────────────────────────────────────────────────────────────
print("-> Cargando modelos base...")

rdv = network.Rendezvous('resnet18', hr_output=False, use_ln=False).to(device)
ckpt_rdv_data = torch.load(CKPT_RDV, map_location=device, weights_only=False)
rdv.load_state_dict(ckpt_rdv_data.get('model_state_dict', ckpt_rdv_data))

tcn = CirugiaTCN().to(device)
ckpt_tcn_data = torch.load(CKPT_TCN, map_location=device, weights_only=False)
tcn.load_state_dict(ckpt_tcn_data.get('model_state_dict', ckpt_tcn_data))

print("-> Ensamblando modelo unificado...")

# ¡AQUÍ ESTÁ LA LÍNEA CORREGIDA!
modelo = SpatioTemporalRDV(rdv, tcn).to(device)

if os.path.exists(CKPT_FUSION):
    fusion_ckpt = torch.load(CKPT_FUSION, map_location=device, weights_only=False)
    # Extraer específicamente el estado del fusion_head
    modelo.refine_inst.load_state_dict({k.replace('refine_inst.', ''): v for k, v in fusion_ckpt['fusion_head_state'].items() if 'refine_inst' in k})
    modelo.refine_verb.load_state_dict({k.replace('refine_verb.', ''): v for k, v in fusion_ckpt['fusion_head_state'].items() if 'refine_verb' in k})
    modelo.refine_targ.load_state_dict({k.replace('refine_targ.', ''): v for k, v in fusion_ckpt['fusion_head_state'].items() if 'refine_targ' in k})
    print("   [+] Cabeza de fusión conectada exitosamente.")
else:
    print("   [!] Sin checkpoint de fusión. Usando pesos iniciales (cuidado).")

modelo.eval()

# ── Preparar Extracción de Datos ───────────────────────────────────────────────
print(f"-> Analizando {NOMBRE_VIDEO}...")
features_np  = np.load(FEATURES_NPY).astype(np.float32)   # [T, 512]
features_seq = torch.from_numpy(features_np.transpose(1, 0)).unsqueeze(0).to(device)  # [1, 512, T]

# Ajusta el umbral. Si dibuja muchas cajas falsas, súbelo a 0.5. Si no dibuja nada, bájalo a 0.15.
extractor_bb = BoundingBoxExtractor(confidence_threshold=0.3)

import glob
rutas_frames = sorted(glob.glob(os.path.join(FRAMES_DIR, "*.*")))
num_frames = min(len(rutas_frames), features_seq.shape[2])

print(f"   Procesando {num_frames} frames combinados...")

with torch.no_grad():
    for frame_idx in range(num_frames):
        ruta_frame = rutas_frames[frame_idx]

        # 1. Preparar visualización
        img_original = cv2.imread(ruta_frame)
        if img_original is None: continue
        img_original = cv2.resize(img_original, (224, 224))
        frame_viz = img_original.copy()

        # 2. Inferencia de red
        with Image.open(ruta_frame) as img:
            frame_tensor = TRANSFORM(img.convert('RGB')).unsqueeze(0).to(device)
            
        salidas = modelo(frame_tensor, features_seq, frame_idx)

        # 3. Dibujar Bounding Boxes y Heatmaps (Solo de Instrumentos para no saturar la imagen)
        bboxes_inst = extractor_bb.extraer(salidas['heatmap_inst'][0], NOMBRES_INST)

        if bboxes_inst:
            # Superponer heatmap
            heatmap_suma = salidas['heatmap_inst'][0].sum(dim=0).cpu().numpy()
            if heatmap_suma.max() > 0:
                heatmap_norm = (heatmap_suma - heatmap_suma.min()) / (heatmap_suma.max() - heatmap_suma.min())
                heatmap_color = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
                frame_viz = cv2.addWeighted(frame_viz, 0.6, heatmap_color, 0.4, 0)

            # Dibujar rectángulos
            for det in bboxes_inst:
                x1, y1, x2, y2 = det['bbox']
                cv2.rectangle(frame_viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame_viz, f"{det['clase']} ({det['confianza']:.2f})", (x1, max(y1-5, 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # 4. Guardar archivo
        nombre_salida = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(nombre_salida, frame_viz)

        if (frame_idx + 1) % 100 == 0:
            print(f"   [{frame_idx+1}/{num_frames}] frames procesados...")

print(f"\n[OK] Frames anotados de {NOMBRE_VIDEO} exportados exitosamente a:\n {OUTPUT_DIR}")