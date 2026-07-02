#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARQUITECTURA TEMPORAL: TCN (Temporal Convolutional Network)
Módulo para el procesamiento de secuencias 1D en reconocimiento
de acciones quirúrgicas — CholecT50.

Nota de diseño: Se usa padding bidireccional (no causal estricto)
porque los videos de CholecT50 son grabaciones offline. Esto maximiza
el contexto temporal para suavizar el flickering del modelo espacial.
Para una implementación en tiempo real futura, cambiar a padding causal.
"""

import torch
import torch.nn as nn


class BloqueTemporal(nn.Module):
    """
    Bloque residual de una TCN con convoluciones 1D dilatadas y BatchNorm.
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, padding, dropout=0.2):
        super(BloqueTemporal, self).__init__()

        self.red_neuronal = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size,
                      stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size,
                      stride=1, padding=padding, dilation=dilation),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Proyección residual si los canales cambian
        self.downsample = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                           if in_channels != out_channels else None)
        self.relu_final = nn.ReLU()

    def forward(self, x):
        salida = self.red_neuronal(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu_final(salida + residual)


class CirugiaTCN(nn.Module):
    """
    Red Convolucional Temporal multietiqueta para CholecT50.

    Entrada:  [Batch, 512, T]  — embeddings espaciales por frame
    Salidas:  4 tensores [Batch, N_clases, T] con logits sin activar
              (el Sigmoid se aplica en la función de pérdida)

    Campo receptivo con config por defecto (4 bloques, kernel=3):
        Bloque 0: dilatación 1 → ve  3 frames
        Bloque 1: dilatación 2 → ve  5 frames
        Bloque 2: dilatación 4 → ve  9 frames
        Bloque 3: dilatación 8 → ve 17 frames
        Campo receptivo total  → 17 frames ≈ 17 segundos a 1fps
    """
    def __init__(self,
                 num_inputs=512,
                 num_channels=None,
                 kernel_size=3,
                 dropout=0.3):
        super(CirugiaTCN, self).__init__()

        if num_channels is None:
            num_channels = [256, 256, 256, 256]

        # Construir bloques con dilatación exponencial
        layers = []
        for i, out_channels in enumerate(num_channels):
            dilation  = 2 ** i
            in_ch     = num_inputs if i == 0 else num_channels[i - 1]
            padding   = (kernel_size - 1) * dilation // 2  # bidireccional
            layers.append(
                BloqueTemporal(in_ch, out_channels, kernel_size,
                               dilation=dilation, padding=padding,
                               dropout=dropout)
            )

        self.extractor_secuencial = nn.Sequential(*layers)

        # Cabezales de predicción (Conv1d 1×1 = Linear aplicado a cada frame)
        tcn_out = num_channels[-1]
        self.head_instrument = nn.Conv1d(tcn_out, 6,   kernel_size=1)
        self.head_verb       = nn.Conv1d(tcn_out, 10,  kernel_size=1)
        self.head_target     = nn.Conv1d(tcn_out, 15,  kernel_size=1)
        self.head_triplet    = nn.Conv1d(tcn_out, 100, kernel_size=1)

    def forward(self, x):
        """
        x: [Batch, 512, T]
        returns: (inst, verb, target, triplet)
                 cada uno de shape [Batch, N_clases, T]
        """
        feats = self.extractor_secuencial(x)   # [Batch, 256, T]

        return (
            self.head_instrument(feats),        # [Batch,   6, T]
            self.head_verb(feats),              # [Batch,  10, T]
            self.head_target(feats),            # [Batch,  15, T]
            self.head_triplet(feats),           # [Batch, 100, T]
        )


# ── Verificación rápida de shapes ────────────────────────────────────────────
if __name__ == "__main__":
    modelo = CirugiaTCN()
    x_prueba = torch.randn(2, 512, 64)   # 2 videos, 512 canales, 64 frames
    inst, verb, targ, trip = modelo(x_prueba)

    print("Verificación de shapes:")
    print(f"  Entrada:      {list(x_prueba.shape)}")
    print(f"  Instrumentos: {list(inst.shape)}")
    print(f"  Verbos:       {list(verb.shape)}")
    print(f"  Órganos:      {list(targ.shape)}")
    print(f"  Tripletes:    {list(trip.shape)}")
    assert inst.shape == (2, 6,   64), "Error en cabezal de instrumentos"
    assert verb.shape == (2, 10,  64), "Error en cabezal de verbos"
    assert targ.shape == (2, 15,  64), "Error en cabezal de órganos"
    assert trip.shape == (2, 100, 64), "Error en cabezal de tripletes"
    print("\n  Todos los shapes son correctos.")