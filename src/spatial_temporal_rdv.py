#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SpatioTemporalRDV — Arquitectura de Fusión Espacial-Temporal
Combina el Rendezvous espacial (CAMs 7×7) con la TCN temporal
para producir heatmaps refinados y bounding boxes por clase.

Flujo:
    Frame(s) → Rendezvous (frozen) → CAMs [B, C, 7, 7]
    .npy     → TCN       (frozen) → Pesos [B, C, T]
                                         ↓
                         CAM × Peso temporal → Heatmap refinado
                                         ↓
                         Upscale 7×7 → 224×224
                                         ↓
                         Otsu threshold → Bounding Box
"""

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# MÓDULO DE REFINAMIENTO ESPACIAL
# Aprende a combinar la CAM espacial con el peso temporal de la TCN.
# Es el único módulo con parámetros entrenables en toda la arquitectura.
# ─────────────────────────────────────────────────────────────────────────────
class SpatialRefinementModule(nn.Module):
    """
    Fusiona una CAM espacial [B, C, 7, 7] con un peso temporal escalar
    [B, C] usando una convolución aprendida + conexión residual.

    El peso temporal actúa como una "máscara de confianza": si la TCN
    dice que el instrumento X tiene 90% de probabilidad en el frame t,
    la CAM de X se amplifica. Si dice 5%, se suprime.
    """
    def __init__(self, num_classes: int, cam_h: int = 7, cam_w: int = 7):
        super().__init__()
        self.num_classes = num_classes
        self.cam_h       = cam_h
        self.cam_w       = cam_w

        # Conv 1×1 para mezclar la CAM modulada con la CAM original
        # Entrada: CAM_modulada concatenada con CAM_original → 2*C canales
        # Salida:  C canales refinados
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
        )

        # Conexión residual: aprende cuánto de la CAM original conservar
        self.residual_weight = nn.Parameter(torch.ones(1))
        self.relu_final      = nn.ReLU(inplace=True)

    def forward(self, cam: torch.Tensor,
                tcn_logits: torch.Tensor) -> torch.Tensor:
        """
        cam        : [B, C, 7, 7]   — CAM del Rendezvous para este frame
        tcn_logits : [B, C]         — logits de la TCN para este frame (sin sigmoid)
        returns    : [B, C, 7, 7]   — CAM refinada
        """
        # Convertir logits TCN a probabilidades [0,1] y expandir a forma espacial
        tcn_weights = torch.sigmoid(tcn_logits)            # [B, C]
        tcn_weights = tcn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Modular la CAM con el peso temporal
        cam_modulada = cam * tcn_weights                   # [B, C, 7, 7]

        # Fusionar CAM modulada con CAM original (concat por canales)
        x = torch.cat([cam_modulada, cam], dim=1)          # [B, 2C, 7, 7]
        x = self.fusion_conv(x)                            # [B, C, 7, 7]

        # Residual: CAM original escalada + refinamiento aprendido
        cam_refinada = self.relu_final(x + self.residual_weight * cam)

        return cam_refinada                                # [B, C, 7, 7]


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITECTURA INTEGRADORA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────
class SpatioTemporalRDV(nn.Module):
    """
    Arquitectura de fusión espacial-temporal para CholecT50.

    Parámetros entrenables: SOLO los SpatialRefinementModules (~50K params).
    Rendezvous y TCN permanecen congelados durante el entrenamiento.

    Args:
        rendezvous  : modelo Rendezvous ya cargado con sus pesos
        tcn         : modelo CirugiaTCN ya cargado con sus pesos
        output_size : resolución del heatmap de salida (default: 224×224)
    """
    def __init__(self, rendezvous: nn.Module, tcn: nn.Module,
                 output_size: int = 224):
        super().__init__()

        self.output_size = output_size

        # ── Modelos congelados ────────────────────────────────────────────────
        self.rendezvous = rendezvous
        self.tcn        = tcn
        self._congelar(self.rendezvous)
        self._congelar(self.tcn)

        # ── Módulos de refinamiento (únicos parámetros entrenables) ──────────
        # Un módulo por rama, con sus dimensiones de clase correspondientes
        self.refine_inst = SpatialRefinementModule(num_classes=6)
        self.refine_verb = SpatialRefinementModule(num_classes=10)
        self.refine_targ = SpatialRefinementModule(num_classes=15)

        # Cabezal final para el triplete: proyecta las 3 CAMs refinadas
        # a 100 clases de triplete (6+10+15 = 31 canales de entrada)
        self.triplet_head = nn.Sequential(
            nn.Conv2d(31, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 100, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
        )

        print(f"[SpatioTemporalRDV] Parámetros entrenables: "
              f"{sum(p.numel() for p in self.parameters() if p.requires_grad):,}")

    @staticmethod
    def _congelar(modelo: nn.Module):
        """Congela todos los parámetros de un modelo."""
        for param in modelo.parameters():
            param.requires_grad = False
        modelo.eval()

    def forward(self,
            frame: torch.Tensor,
            features_secuencia: torch.Tensor,
            frame_idx: int) -> dict:
        """
        frame              : [1, 3, 224, 224]
        features_secuencia : [1, 512, T]
        frame_idx          : int
        """
    # ── 1. Extracción espacial (Rendezvous) ───────────────────────────────
    # Sin torch.no_grad() aquí — los parámetros congelados no generan
    # gradientes por sí solos (requires_grad=False), pero el grafo
    # debe fluir hasta los SpatialRefinementModules que SÍ son entrenables
        enc_i, enc_v, enc_t, dec_ivt = self.rendezvous(frame)

        cam_i = enc_i[0]   # [1, 6,  7, 7]
        cam_v = enc_v[0]   # [1, 10, 7, 7]
        cam_t = enc_t[0]   # [1, 15, 7, 7]

    # ── 2. Extracción temporal (TCN) ──────────────────────────────────────
    # Aquí SÍ usamos no_grad porque la salida de la TCN se usa como
    # peso escalar — no necesita grafo, solo el valor numérico
        with torch.no_grad():
            tcn_i, tcn_v, tcn_t, tcn_ivt = self.tcn(features_secuencia)

    # Extraer frame_idx de la dimensión temporal → [1, C]
    # detach() porque este tensor viene de no_grad y no debe
    # intentar propagarse hacia atrás
        peso_i = tcn_i[:, :, frame_idx].detach()   # [1, 6]
        peso_v = tcn_v[:, :, frame_idx].detach()   # [1, 10]
        peso_t = tcn_t[:, :, frame_idx].detach()   # [1, 15]

    # ── 3. Refinamiento espacial (parámetros entrenables) ─────────────────
    # El grafo fluye desde cam_i/v/t → SpatialRefinementModule → loss
        cam_i_ref = self.refine_inst(cam_i, peso_i)   # [1, 6,  7, 7]
        cam_v_ref = self.refine_verb(cam_v, peso_v)   # [1, 10, 7, 7]
        cam_t_ref = self.refine_targ(cam_t, peso_t)   # [1, 15, 7, 7]

    # ── 4. Cabezal de triplete ────────────────────────────────────────────
        cam_concat  = torch.cat([cam_i_ref, cam_v_ref, cam_t_ref], dim=1)
        cam_triplet = self.triplet_head(cam_concat)

    # ── 5. Upscale a 224×224 ─────────────────────────────────────────────
        def upscale(cam):
         return F.interpolate(cam, size=(self.output_size, self.output_size),
                             mode='bilinear', align_corners=False)

        heatmap_i   = upscale(cam_i_ref)
        heatmap_v   = upscale(cam_v_ref)
        heatmap_t   = upscale(cam_t_ref)
        heatmap_ivt = upscale(cam_triplet)

        return {
            "heatmap_inst"    : heatmap_i,
            "heatmap_verb"    : heatmap_v,
            "heatmap_target"  : heatmap_t,
            "heatmap_triplet" : heatmap_ivt,
            "cam_inst_raw"    : cam_i_ref,
            "cam_verb_raw"    : cam_v_ref,
            "cam_target_raw"  : cam_t_ref,
            "logits_inst"     : enc_i[1],
            "logits_verb"     : enc_v[1],
            "logits_target"   : enc_t[1],
            "logits_triplet"  : dec_ivt,
    }


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTOR DE BOUNDING BOXES (sin entrenamiento, puro OpenCV)
# ─────────────────────────────────────────────────────────────────────────────
class BoundingBoxExtractor:
    """
    Convierte un heatmap [C, H, W] en bounding boxes por clase
    usando threshold de Otsu + detección de contornos.

    No tiene parámetros aprendibles. Es un postprocesador determinista.
    """
    def __init__(self,
                 confidence_threshold: float = 0.3,
                 min_area_ratio: float = 0.001):
        """
        confidence_threshold : descarta clases con activación máxima menor a este valor
        min_area_ratio       : descarta contornos menores a este % del área total
        """
        self.conf_thresh  = confidence_threshold
        self.min_area_ratio = min_area_ratio

    def extraer(self, heatmap: torch.Tensor,
                nombres_clases: list) -> list:
        """
        heatmap        : [C, H, W] tensor (una sola muestra, sin batch dim)
        nombres_clases : lista de strings con el nombre de cada clase

        returns: lista de dicts con las detecciones encontradas
        [
          {
            'clase'     : 'Grasper',
            'clase_idx' : 0,
            'confianza' : 0.87,
            'bbox'      : (x1, y1, x2, y2),   # píxeles en imagen 224×224
            'centro'    : (cx, cy),
          },
          ...
        ]
        """
        detecciones = []
        heatmap_np  = heatmap.detach().cpu().numpy()  # [C, H, W]
        H, W        = heatmap_np.shape[1], heatmap_np.shape[2]
        area_total  = H * W

        for clase_idx in range(heatmap_np.shape[0]):
            mapa = heatmap_np[clase_idx]               # [H, W]

            # Normalizar a [0, 255] para OpenCV
            mapa_min, mapa_max = mapa.min(), mapa.max()
            if mapa_max - mapa_min < 1e-6:
                continue   # mapa vacío, no hay activación

            confianza = float(mapa_max)
            if confianza < self.conf_thresh:
                continue   # clase no activa en este frame

            mapa_norm = ((mapa - mapa_min) /
                         (mapa_max - mapa_min) * 255).astype(np.uint8)

            # Threshold de Otsu (automático, sin parámetro manual)
            _, binario = cv2.threshold(
                mapa_norm, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            # Encontrar contornos
            contornos, _ = cv2.findContours(
                binario, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for contorno in contornos:
                area = cv2.contourArea(contorno)
                if area / area_total < self.min_area_ratio:
                    continue   # ruido, área demasiado pequeña

                x, y, w, h = cv2.boundingRect(contorno)
                detecciones.append({
                    'clase'     : nombres_clases[clase_idx],
                    'clase_idx' : clase_idx,
                    'confianza' : round(confianza, 4),
                    'bbox'      : (x, y, x + w, y + h),
                    'centro'    : (x + w // 2, y + h // 2),
                })

        return sorted(detecciones, key=lambda d: d['confianza'], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────
# VERIFICACIÓN RÁPIDA DE SHAPES
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys, os, warnings
    warnings.filterwarnings("ignore")

    root = r"C:\\ProyectoRDV"
    sys.path.append(root)

    import src.network as network
    from src.tcn import CirugiaTCN

    # Cargar modelos reales
    rdv = network.Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()
    ckpt = torch.load(os.path.join(root, "checkpoints", "run_fusion",
                                   "Rendezvous_FUSION_FINAL.pth"),
                      map_location='cuda:0', weights_only=False)
    rdv.load_state_dict(ckpt.get('model_state_dict', ckpt))

    tcn = CirugiaTCN().cuda()
    tcn_ckpt = torch.load(os.path.join(root, "checkpoints", "TCN_FINAL.pth"),
                          map_location='cuda:0', weights_only=False)
    tcn.load_state_dict(tcn_ckpt.get('model_state_dict', tcn_ckpt))

    # Instanciar arquitectura fusionada
    modelo = SpatioTemporalRDV(rdv, tcn).cuda()
    modelo.eval()

    # Datos ficticios
    frame_fake    = torch.randn(1, 3, 224, 224).cuda()
    seq_fake      = torch.randn(1, 512, 64).cuda()

    with torch.no_grad():
        salidas = modelo(frame_fake, seq_fake, frame_idx=10)

    print("=" * 55)
    print("  VERIFICACIÓN — SpatioTemporalRDV")
    print("=" * 55)
    for nombre, tensor in salidas.items():
        print(f"  {nombre:<22}: {list(tensor.shape)}")

    # Probar extractor de bounding boxes
    nombres_inst = ['Grasper','Bipolar','Hook','Scissors','Clipper','Irrigator']
    extractor    = BoundingBoxExtractor(confidence_threshold=0.1)
    bboxes       = extractor.extraer(
        salidas['heatmap_inst'][0],   # [6, 224, 224]
        nombres_inst
    )

    print("\n  BOUNDING BOXES detectadas (datos ficticios):")
    if bboxes:
        for b in bboxes:
            print(f"    {b['clase']:<12} conf:{b['confianza']:.3f}"
                  f"  bbox:{b['bbox']}  centro:{b['centro']}")
    else:
        print("    (ninguna sobre el umbral con datos ficticios)")
    print("=" * 55)
