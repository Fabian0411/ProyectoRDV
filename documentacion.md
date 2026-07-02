# SpatioTemporalRDV
### Implementación Modular de la Arquitectura Rendezvous con Fusión Espaciotemporal para el Reconocimiento de Tripletes Quirúrgicos en Colecistectomía Laparoscópica

**Fabián Ortiz Carreño** · Ingeniería Mecatrónica · Facultad de Ingeniería, UNAM  
Servicio Social — Semestre 2026-2 · Supervisor: Dr. Daniel Haro Mendoza

---

## Descripción

Este repositorio implementa una arquitectura de reconocimiento de tripletes quirúrgicos `<Instrumento, Verbo, Órgano>` sobre el dataset [CholecT50](https://github.com/CAMMA-public/cholect50), adaptada para entrenamiento en hardware de consumo (GPU con ≥3 GB de VRAM).

La contribución principal es la arquitectura **SpatioTemporalRDV**: una cabeza de fusión aprendible que combina mapas de activación (CAMs) del modelo Rendezvous espacial con pesos temporales de una Red Convolucional Temporal (TCN), produciendo predicciones más estables y precisas que el modelo puramente espacial.

### Resultados principales (Fold 1, cholect50-crossval)

| Métrica | RDV Original [1] | RDV Espacial (este trabajo) | SpatioTemporalRDV (este trabajo) |
|---|---|---|---|
| AP Instrumentos | 92.00 % | 70.78 % | 77.56 % |
| AP Verbos | 60.70 % | 48.70 % | 53.59 % |
| AP Órganos | 38.30 % | 18.10 % | **38.45 %** ↑ |
| mAP Triplete | 29.90 % | 2.06 % | 13.38 % |
| Hardware | Clúster multi-GPU | GTX 1050 (3 GB) | GTX 1050 (3 GB) |

> **Nota metodológica:** Los resultados del modelo original corresponden al promedio de 5-fold cross-validation completo. Los resultados de este trabajo corresponden únicamente al Fold 1, por lo que la comparación es orientativa y no constituye una réplica del protocolo completo.

---

## Arquitectura del pipeline

```
FASE A — Entrenamiento Espacial (Rendezvous)
  Frame RGB → ResNet-18 → WSL / CAGAM / T-CAM → Logits por componente

FASE B — Extracción offline de características
  Frame RGB → ResNet-18 (congelada) → hook en layer4[1].bn2 → vector [512] → .npy

FASE C — Entrenamiento TCN
  Secuencia [T, 512] → TCN (4 bloques dilatados) → logits temporales por frame

FASE D — Extracción offline de CAMs
  Frame RGB → Rendezvous (congelado) → enc_i[0], enc_v[0], enc_t[0] → CAMs [C, 7, 7] → .npy

FASE E — Entrenamiento de la Cabeza de Fusión
  CAMs [B, C, 7, 7] + Pesos TCN [B, C] → SpatialRefinementModule → logits refinados
```

---

## Estructura del repositorio

```
SpatioTemporalRDV/
│
├── data/                          ← NO se sube a GitHub (ver .gitignore)
│   ├── CholecT50/
│   │   ├── videos/
│   │   │   ├── VID01/             ← frames .png del video (000000.png, 000001.png, ...)
│   │   │   ├── VID02/
│   │   │   └── ...
│   │   ├── triplet/               ← etiquetas de tripletes (VID01.txt, VID02.txt, ...)
│   │   ├── instrument/            ← etiquetas de instrumentos (.txt)
│   │   ├── verb/                  ← etiquetas de verbos (.txt)
│   │   ├── target/                ← etiquetas de órganos (.txt)
│   │   └── labels/                ← anotaciones en formato JSON (VID01.json, ...)
│   │
│   ├── features_1d_custom/        ← vectores [T, 512] extraídos por video (.npy)
│   └── cams_offline/              ← CAMs pre-calculadas por video
│       ├── cams_inst_VID01.npy    ← shape [T, 6,  7, 7]
│       ├── cams_verb_VID01.npy    ← shape [T, 10, 7, 7]
│       ├── cams_targ_VID01.npy    ← shape [T, 15, 7, 7]
│       └── ...
│
├── checkpoints/                   ← NO se sube a GitHub (ver .gitignore)
│   ├── run_instrumentos/
│   │   └── Rendezvous_ESPECIALISTA_INSTRUMENTOS.pth
│   ├── run_verbos/
│   │   └── Rendezvous_ESPECIALISTA_VERBOS.pth
│   ├── run_organos/
│   │   └── Rendezvous_ESPECIALISTA_ORGANOS.pth
│   ├── run_fusion/
│   │   └── Rendezvous_FUSION_FINAL.pth     ← modelo espacial final (Época 21)
│   ├── TCN_FINAL.pth                        ← TCN entrenada (Época 7)
│   ├── SpatioTemporal_FINAL.pth             ← cabeza de fusión entrenada (Época 50)
│   ├── TCN_training.log
│   └── fusion_training_v4.log
│
├── src/                           ← código fuente
│   ├── __init__.py
│   ├── network.py                 ← arquitectura Rendezvous (ResNet-18 + WSL + CAGAM + MHMA)
│   ├── dataloader.py              ← dataset CholecT50 (clase T50 y CholecT50)
│   └── tcn.py                     ← arquitectura TCN (bloques dilatados 1D)
│
├── scripts/                       ← scripts ejecutables en orden de pipeline
│   ├── run_instrumentos.py        ← Fase A.1: entrena especialista de instrumentos
│   ├── run_verbos.py              ← Fase A.2: entrena especialista de verbos
│   ├── run_organos.py             ← Fase A.3: entrena especialista de órganos
│   ├── run_fusion.py              ← Fase A.4: fusión end-to-end de especialistas
│   ├── extract_features.py        ← Fase B: extrae vectores 1D a features_1d_custom/
│   ├── train_temporal.py          ← Fase C: entrena la TCN
│   ├── extract_cams.py            ← Fase D: pre-calcula CAMs a cams_offline/
│   ├── train_fusion_head.py       ← Fase E: entrena la cabeza de fusión
│   ├── evaluador_rdv_puro.py      ← evaluación del modelo espacial con ivtmetrics
│   └── evaluador_map.py           ← evaluación del SpatioTemporalRDV con ivtmetrics
│
├── plots/                         ← gráficas y matrices de confusión generadas
│   ├── especialistas/
│   └── fusion/
│
├── legacy/                        ← experimentos descartados y archivos históricos
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Requisitos

### Hardware mínimo
- GPU NVIDIA con ≥ 3 GB de VRAM (probado en GTX 1050 3 GB)
- RAM del sistema: ≥ 16 GB recomendado (el dataset completo cargado en RAM ocupa ~8 GB)
- Almacenamiento: ~50 GB para frames de video + ~2 GB para CAMs pre-calculadas

### Software
```
Python          3.11
PyTorch         2.6.0+cu124
torchvision     0.21.0+cu124
numpy           2.4.x
pillow          12.x
opencv-python   4.13.x
ivtmetrics      0.1.5
```

### Instalación
```bash
git clone https://github.com/TU_USUARIO/SpatioTemporalRDV.git
cd SpatioTemporalRDV

# Crear entorno virtual
python -m venv venv --system-site-packages
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Instalar PyTorch con CUDA (ajusta la versión según tu driver)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Instalar dependencias restantes
pip install numpy pillow opencv-python ivtmetrics
```

Verificar que CUDA está disponible:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
# Debe imprimir: True y el nombre de tu GPU
```

---

## Dataset

Solicita acceso al dataset CholecT50 en: https://github.com/CAMMA-public/cholect50

Una vez descargado, organiza los archivos según la estructura indicada en la sección anterior. Los frames de cada video deben estar en `data/CholecT50/videos/VIDXX/` nombrados como `000000.png`, `000001.png`, etc.

---

## Pipeline de entrenamiento completo

Sigue las fases en orden. Cada fase depende de los artefactos producidos por la anterior.

---

### FASE A — Entrenamiento del modelo Rendezvous espacial

Esta fase entrena tres "especialistas" por separado y luego los fusiona en un único modelo end-to-end. El objetivo es evitar la competencia de gradientes que ocurre en el entrenamiento monolítico cuando las señales de instrumentos dominan sobre las de órganos.

**A.1 — Especialista de Instrumentos**
```bash
cd scripts
python run_instrumentos.py --train -b 4 --epochs 8 --data_dir ../data/CholecT50
```
Checkpoint guardado en: `checkpoints/run_instrumentos/Rendezvous_ESPECIALISTA_INSTRUMENTOS.pth`  
Rendimiento esperado: mAP ≈ 86 % en validación (época 6).

**A.2 — Especialista de Órganos**
```bash
python run_organos.py --train -b 4 --epochs 10 --data_dir ../data/CholecT50
```
Checkpoint guardado en: `checkpoints/run_organos/Rendezvous_ESPECIALISTA_ORGANOS.pth`  
Rendimiento esperado: mAP_T ≈ 27 % en validación (época 9).

**A.3 — Especialista de Verbos**
```bash
python run_verbos.py --train -b 4 --epochs 15 --data_dir ../data/CholecT50
```
Checkpoint guardado en: `checkpoints/run_verbos/Rendezvous_ESPECIALISTA_VERBOS.pth`  
Rendimiento esperado: mAP_V ≈ 54 % en validación (época 14).

**A.4 — Fusión end-to-end**

Inyecta los pesos de los tres especialistas en un único modelo y aplica ajuste fino con learning rate microscópico.

```bash
python run_fusion.py --train -b 4 --epochs 30 \
  --data_dir ../data/CholecT50 \
  --dataset_variant cholect45-crossval \
  --kfold 1
```
Checkpoint guardado en: `checkpoints/run_fusion/Rendezvous_FUSION_FINAL.pth`  
El script carga automáticamente los especialistas de órganos y verbos desde sus rutas en `__checkpoint__/`.

> **Nota:** Esta fase puede tomar varias horas en hardware de consumo. Se incluye reanudación automática: si el proceso se interrumpe, volver a ejecutar el mismo comando continuará desde el último checkpoint guardado.

---

### FASE B — Extracción offline de vectores 1D (para la TCN)

Pasa cada frame por la ResNet-18 del modelo de fusión y guarda el vector resultante del pooling global. Esto se hace **una sola vez** y permite entrenar la TCN sin requerir las imágenes originales.

```bash
python extract_features.py
```

El script:
1. Carga `Rendezvous_FUSION_FINAL.pth` y conecta un hook en `encoder.basemodel.basemodel.layer4[1].bn2`.
2. Aplica un `AdaptiveAvgPool2d(1,1)` manual para obtener vectores `[512]` por frame.
3. Guarda un archivo `.npy` de shape `[T, 512]` por video en `data/features_1d_custom/`.

Tiempo estimado: ~2–3 horas para los 50 videos de CholecT50 en GTX 1050.

---

### FASE C — Entrenamiento de la TCN

Entrena la Red Convolucional Temporal sobre las secuencias de vectores 1D extraídas en la Fase B.

```bash
python train_temporal.py
```

**Parámetros de la TCN:**

| Parámetro | Valor |
|---|---|
| Bloques | 4 |
| Canales por bloque | 256 |
| Kernel size | 3 |
| Dilataciones | 1 → 2 → 4 → 8 |
| Campo receptivo | 17 frames |
| Tipo de padding | Bidireccional (offline) |
| Normalización | BatchNorm1D |
| Dropout | 0.3 |
| LR | 1e-4 |
| Early stopping | patience = 10 épocas |
| Semilla aleatoria | 42 |

El script realiza validación cruzada 80/20 por video con semilla fija. El checkpoint se guarda en `checkpoints/TCN_FINAL.pth` únicamente cuando mejora la `val_loss`.

Rendimiento esperado: mejor `val_loss` ≈ 0.45 (BCEWithLogitsLoss, época ~7–9).

---

### FASE D — Extracción offline de CAMs

Pre-calcula los mapas de activación por clase del modelo Rendezvous para cada frame y los guarda en disco. Esto permite entrenar la cabeza de fusión sin procesar imágenes en cada época.

```bash
python extract_cams.py
```

El script genera por cada video tres archivos en `data/cams_offline/`:
- `cams_inst_VIDXX.npy` — shape `[T, 6, 7, 7]`
- `cams_verb_VIDXX.npy` — shape `[T, 10, 7, 7]`
- `cams_targ_VIDXX.npy` — shape `[T, 15, 7, 7]`

Tamaño aproximado: ~2 MB por archivo, ~300 MB para los 50 videos.  
Tiempo estimado: ~2–3 horas (misma magnitud que la Fase B).

> **Por qué es necesario:** Las CAMs del Rendezvous tienen información espacial 7×7 que el Global Average Pooling de la Fase B destruye. Para la cabeza de fusión necesitamos ese mapa espacial intacto.

---

### FASE E — Entrenamiento de la Cabeza de Fusión

Entrena únicamente los 28,482 parámetros del `FusionHeadTrainer`. Los modelos Rendezvous y TCN permanecen congelados; sus artefactos (CAMs y pesos temporales) se leen de disco y RAM respectivamente.

```bash
python train_fusion_head.py
```

El dataset completo (~100K frames × CAMs + logits TCN + etiquetas) se carga en RAM al inicio (~1 minuto). Cada época toma aproximadamente 22 segundos.

**Parámetros de entrenamiento:**

| Parámetro | Valor |
|---|---|
| LR | 1e-4 |
| Batch size | 32 |
| Épocas máximas | 50 |
| Early stopping | patience = 10 |
| Semilla aleatoria | 42 |
| Normalización de CAMs | `(x - mean) / (std + 1e-6)` por video |
| Función de pérdida | BCEWithLogitsLoss |
| Optimizador | Adam |

> **Corrección crítica implementada:** El `SpatialRefinementModule` **no tiene ReLU al final del residual**. Las CAMs del Rendezvous tienen valores mayoritariamente negativos (media ≈ −6.9), y añadir ReLU al final de la suma residual produce salidas nulas, estancando la pérdida en ln(2) ≈ 0.693. Esta corrección redujo la pérdida de validación de 3.38 a 0.40 en 50 épocas.

Checkpoint guardado en: `checkpoints/SpatioTemporal_FINAL.pth`

---

## Evaluación

### Evaluar el Rendezvous espacial puro
```bash
python evaluador_rdv_puro.py
```
Evalúa `Rendezvous_FUSION_FINAL.pth` sobre los 9 videos del Fold 1 (cholect45-crossval) usando `ivtmetrics`.

### Evaluar el SpatioTemporalRDV completo
```bash
python evaluador_map.py
```
Evalúa la cabeza de fusión entrenada con CAMs offline y pesos TCN pre-calculados sobre los 10 videos del Fold 1 (cholect50-crossval) usando `ivtmetrics`.

Ambos evaluadores generan salida en consola con AP desglosado por componente y mAP global del triplete.

---

## Notas de reproducibilidad

- Los experimentos se realizaron en **Windows 11** con **Python 3.11** y **PyTorch 2.6.0+cu124**.
- La semilla aleatoria `42` se fijó en los splits de datos. Los pesos iniciales de las redes neuronales no tienen semilla fija, por lo que puede haber varianza entre corridas.
- El entrenamiento completo (Fases A–E) toma aproximadamente 15–20 horas en una GTX 1050 de 3 GB VRAM.
- El repositorio del modelo Rendezvous original está disponible en: https://github.com/CAMMA-public/rendezvous

---

## Referencias

```
[1] C. I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux y N. Padoy,
    "Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic
    Videos," Medical Image Analysis, vol. 78, p. 102433, 2022.
    DOI: 10.1016/j.media.2022.102433

[2] C. I. Nwoye et al., "CholecTriplet2021: A benchmark challenge for surgical action triplet
    recognition," Medical Image Analysis, vol. 89, p. 102897, 2023.
    DOI: 10.1016/j.media.2023.102897
```

---

## Licencia

Este repositorio contiene código de investigación académica sin fines comerciales.  
El código del dataloader (`src/dataloader.py`) está basado en el repositorio original de CAMMA (Apache License 2.0).

---

*Proyecto desarrollado como parte del Servicio Social de Ingeniería Mecatrónica, FI-UNAM, 2026.*
