# SpatioTemporalRDV

### Fusión Espacio-Temporal para Reconocimiento de Tripletes Quirúrgicos `<Instrumento, Verbo, Órgano>` en Hardware de Consumo

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CholecT50-informational?style=flat-square)](https://github.com/CAMMA-public/cholect50)
[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
[![Hardware](https://img.shields.io/badge/Entrenado%20en-GTX%201050%20(3GB%20VRAM)-blue?style=flat-square)](#-instalación-y-uso)

---

##  Descripción

**SpatioTemporalRDV** es una reimplementación modular y una extensión arquitectónica del modelo
**Rendezvous** (Nwoye et al., 2022) para el reconocimiento de tripletes quirúrgicos sobre el dataset
[CholecT50](https://github.com/CAMMA-public/cholect50), diseñada para entrenarse en **GPUs de consumo
con tan solo 3 GB de VRAM** en lugar del clúster multi-GPU usado en el trabajo original.

Entrenar el Rendezvous de forma monolítica en hardware limitado expone dos problemas que degradan
severamente su desempeño:

- **Competencia de gradientes.** Al optimizar instrumentos, verbos y órganos de forma simultánea, la
  señal de instrumentos (frecuente y de alta magnitud) domina el gradiente compartido y ahoga el
  aprendizaje de las señales de verbos y, sobre todo, de órganos —la tarea más difícil y con clases más
  desbalanceadas.
- **El "mar rosa".** En cirugía laparoscópica, la inmensa mayoría de los píxeles de cada frame son
  tejido rosado/rojizo sin información discriminativa; el instrumento u órgano de interés ocupa una
  fracción mínima de la imagen. Esto produce mapas de activación (CAMs) ruidosos que "parpadean"
  (flickering) de un frame a otro, incluso cuando la acción quirúrgica es continua.

**SpatioTemporalRDV** resuelve ambos problemas separando el entrenamiento en fases desacopladas:

1. Se entrenan **especialistas** independientes por componente (instrumento, verbo, órgano) para
   eliminar la competencia de gradientes.
2. Se fusionan en un único Rendezvous con ajuste fino de bajo *learning rate*.
3. Se extraen **offline** las características espaciales y los CAMs de cada frame (una sola pasada,
   sin recalcular nunca las imágenes).
4. Una **Red Convolucional Temporal (TCN)** aprende, a partir de la secuencia completa del video, la
   confianza por clase de cada frame.
5. Una **cabeza de fusión aprendible** modula cada CAM espacial con su confianza temporal
   correspondiente, estabilizando la predicción final del triplete.

El resultado es un pipeline que cabe en 3 GB de VRAM (batch físico de 4 con acumulación de gradiente
×8) y que, pese a esa restricción, **iguala o supera al modelo original en la componente de Órganos**.

---

##  Resultados (Test Fold 1, CholecT50)

| Métrica | RDV Original [1] (5-fold, clúster) | RDV Espacial (este trabajo) | **SpatioTemporalRDV (este trabajo)** |
|---|:---:|:---:|:---:|
| AP Instrumentos | 92.00 % | 70.78 % | **77.56 %** |
| AP Verbos | 60.70 % | 48.70 % | **53.59 %** |
| AP Órganos | 38.30 % | 18.10 % | **38.45 %** |
| mAP Triplete | 29.90 % | 2.06 % | **13.38 %** |
| Hardware | Clúster multi-GPU | GTX 1050 (3 GB) | GTX 1050 (3 GB) |

>  **La cabeza de fusión espacio-temporal supera el estado del arte del artículo original en AP de
> Órganos (38.45 % vs. 38.30 %)**, entrenando en una fracción del hardware usado por los autores
> originales.

> **Nota metodológica:** los resultados de "RDV Original" corresponden al promedio del 5-fold
> cross-validation reportado en el paper. Los resultados de este trabajo corresponden únicamente al
> Fold 1, por lo que la comparación es orientativa y no constituye una réplica del protocolo completo.
> La columna "RDV Espacial" es una ablación: el mismo modelo Rendezvous fusionado, evaluado *sin* la
> TCN ni la cabeza de fusión — cuantifica cuánto aporta la etapa espacio-temporal.

---

##  Estructura del Proyecto

```
SpatioTemporalRDV/
├── src/                      # Código fuente canónico
│   ├── network.py            # Arquitectura Rendezvous (ResNet-18 + WSL + CAGAM + MHMA)
│   ├── dataloader.py         # Dataset CholecT50 y lógica de splits oficiales
│   ├── tcn.py                # Red Convolucional Temporal (CirugiaTCN)
│   └── spatial_temporal_rdv.py  # Módulo de refinamiento espacio-temporal
│
├── scripts/                  # Entry points ejecutables, uno por fase del pipeline
│   ├── run_instrumentos.py / run_organos.py / run_verbos.py   # Fase A: especialistas
│   ├── run_fusion.py                                          # Fase A.4: fusión end-to-end
│   ├── extract_features.py / extract_cams.py                  # Fases B/D: extracción offline
│   ├── train_temporal.py                                      # Fase C: entrenamiento de la TCN
│   ├── train_fusion_head.py                                   # Fase E: cabeza de fusión
│   └── evaluador_rdv_puro.py                                  # Evaluación del modelo espacial
│
├── data/CholecT50/           # Dataset (no versionado — ver .gitignore)
├── docs/                     # Reporte extenso del proyecto
├── checkpoints/              # Pesos entrenados (no versionado — ver .gitignore)
├── plots/                    # Gráficas y matrices de confusión generadas
├── evaluador_map.py          # Evaluación oficial (ivtmetrics) del modelo completo
├── requirements.txt
└── README.md
```

---

##  Instalación y Uso

### Requisitos

| | Mínimo |
|---|---|
| GPU | NVIDIA con ≥ 3 GB de VRAM (probado en GTX 1050 3 GB) |
| RAM | ≥ 16 GB recomendado (el dataset de features/CAMs cargado en RAM ocupa ~8 GB en la Fase E) |
| Disco | ~50 GB para los frames de video + ~2 GB para CAMs pre-calculadas |
| Software | Python 3.11, PyTorch 2.6+ con CUDA, `ivtmetrics==0.1.5` |

### 1. Clonar e instalar dependencias

```bash
git clone https://github.com/<tu-usuario>/SpatioTemporalRDV.git
cd SpatioTemporalRDV

# Entorno virtual
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Dependencias del proyecto
pip install -r requirements.txt

# Si `torch` se instaló sin soporte CUDA, reinstálalo con la rueda correcta para tu driver:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Verifica que CUDA esté disponible antes de continuar:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 2. Preparar el dataset

Solicita acceso a CholecT50 en el [repositorio oficial de CAMMA](https://github.com/CAMMA-public/cholect50)
y colócalo en `data/CholecT50/` siguiendo la estructura del árbol de arriba (`videos/VIDXX/000000.png, ...`,
`triplet/`, `instrument/`, `verb/`, `target/`, `labels/*.json`).

### 3. Pipeline de entrenamiento

Cada fase depende de los artefactos producidos por la anterior — ejecútalas en orden. Todas soportan
**reanudación automática**: si el proceso se interrumpe, volver a correr el mismo comando continúa desde
el último checkpoint guardado.

> ⚠️ `run_instrumentos.py`, `run_organos.py`, `run_verbos.py` y `run_fusion.py` importan `network` y
> `dataloader` sin prefijo de paquete, por lo que `src/` debe estar en el `PYTHONPATH` al ejecutarlos:
> ```bash
> # PowerShell
> $env:PYTHONPATH = "$PWD\src;$env:PYTHONPATH"
> # Linux/Mac
> export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
> ```

**Fase A — Especialistas por componente** (evita la competencia de gradientes)

```bash
cd scripts
python run_instrumentos.py --train -b 4 --epochs 8  --data_dir ../data/CholecT50   # mAP_I ≈ 86 % (época 6)
python run_organos.py      --train -b 4 --epochs 10 --data_dir ../data/CholecT50   # mAP_T ≈ 27 % (época 9)
python run_verbos.py       --train -b 4 --epochs 15 --data_dir ../data/CholecT50   # mAP_V ≈ 54 % (época 14)
```

**Fase A.4 — Fusión end-to-end.** Inyecta los pesos de los tres especialistas en un único Rendezvous y
aplica ajuste fino con un *learning rate* microscópico:

```bash
python run_fusion.py --train -b 4 --epochs 30 \
  --data_dir ../data/CholecT50 \
  --dataset_variant cholect45-crossval \
  --kfold 1
```

Checkpoint resultante: `checkpoints/run_fusion/Rendezvous_FUSION_FINAL.pth`. Esta fase es la más larga
del pipeline (varias horas en hardware de consumo).

**Fase B — Extracción offline de vectores 1D** (para la TCN). Pasa cada frame por la ResNet-18 del
modelo fusionado y guarda el vector post-pooling `[512]`, para no volver a tocar las imágenes originales:

```bash
cd ..
python scripts/extract_features.py    # → data/features_1d_custom/VIDxx.npy  [T, 512]
```

**Fase C — Entrenamiento de la TCN** (4 bloques dilatados, padding bidireccional, campo receptivo de
17 frames):

```bash
python scripts/train_temporal.py      # → checkpoints/TCN_FINAL.pth (val_loss ≈ 0.45, época 7–9)
```

**Fase D — Extracción offline de CAMs.** El *global average pooling* de la Fase B destruye la
información espacial 7×7; esta fase la conserva para la cabeza de fusión:

```bash
python scripts/extract_cams.py        # → data/cams_offline/cams_{inst,verb,targ}_VIDxx.npy
```

**Fase E — Entrenamiento de la cabeza de fusión** (~28K parámetros; Rendezvous y TCN permanecen
congelados, solo se leen sus artefactos de disco/RAM):

```bash
python scripts/train_fusion_head.py   # → checkpoints/SpatioTemporal_FINAL.pth
```

> **Detalle de la corrección clave:** el `SpatialRefinementModule` de esta fase **no aplica ReLU al
> final de la conexión residual**. Las CAMs del Rendezvous son mayoritariamente negativas (media ≈ −6.9);
> un ReLU ahí anulaba la señal y estancaba la pérdida en `ln(2) ≈ 0.693`. Al retirarlo, la pérdida de
> validación bajó de 3.38 a 0.40 en 50 épocas.

### 4. Evaluación

```bash
python scripts/evaluador_rdv_puro.py  # RDV espacial puro, sin TCN ni cabeza de fusión
python evaluador_map.py               # SpatioTemporalRDV completo — genera la tabla de resultados
```

Ambos evaluadores usan `ivtmetrics` (`compute_video_AP`, `ignore_null=False`) sobre los videos de test
del Fold 1 e imprimen el AP desglosado por componente junto al mAP global del triplete.

---

##  Limitaciones conocidas

- El entrenamiento completo (Fases A–E) toma entre 15–20 horas en una GTX 1050 de 3 GB.
- Los resultados corresponden a un único fold (Fold 1); no se realizó el 5-fold cross-validation
  completo del artículo original por restricciones de cómputo.
- Algunos scripts asumen la ruta absoluta del proyecto o requieren `PYTHONPATH` configurado
  manualmente (ver nota en la Fase A) — pendiente de resolver antes de empaquetar como librería.
- Los scripts de inferencia/visualización (`scripts/generar_video.py`, `scripts/inferencia_visual.py`,
  `scripts/inferencia_visualizacion.py`, `generador_matriz*`) no forman parte del pipeline reproducible
  documentado arriba: tienen rutas absolutas incrustadas (algunas apuntando a una carpeta personal de
  OneDrive) y, en el caso de `generar_video.py`, cargan un checkpoint de una corrida antigua
  (`__checkpoint__/run_0/...`) en vez del `Rendezvous_FUSION_FINAL.pth` de la Fase A.4. Útiles como
  referencia interna, no listos para que un tercero los ejecute sin editarlos primero.

---

##  Agradecimientos y Citación

Esta arquitectura extiende el trabajo original de **CAMMA (University of Strasbourg)**. El modelo base
Rendezvous, el dataset CholecT50 y la métrica oficial `ivtmetrics` pertenecen a sus autores originales;
este repositorio únicamente aporta la etapa de fusión espacio-temporal descrita arriba.

```bibtex
@article{nwoye2022rendezvous,
  title   = {Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos},
  author  = {Nwoye, Chinedu Innocent and Yu, Tong and Gonzalez, Cristians and Seeliger, Barbara and Mascagni, Pietro and Mutter, Didier and Marescaux, Jacques and Padoy, Nicolas},
  journal = {Medical Image Analysis},
  volume  = {78},
  pages   = {102433},
  year    = {2022},
  doi     = {10.1016/j.media.2022.102433}
}
```

- Repositorio original del modelo: [CAMMA-public/rendezvous](https://github.com/CAMMA-public/rendezvous)
- Dataset: [CAMMA-public/cholect50](https://github.com/CAMMA-public/cholect50)
- Métrica oficial: [`ivtmetrics`](https://pypi.org/project/ivtmetrics/)

##  Licencia

Código de investigación académica sin fines comerciales, distribuido bajo
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), en línea con la licencia del
modelo y el dataset originales. `src/dataloader.py` está basado en el repositorio original de CAMMA
(Apache License 2.0).

---

**Fabián Ortiz Carreño** · Ingeniería Mecatrónica, Facultad de Ingeniería, UNAM
Servicio Social — Semestre 2026-2 · Supervisor: Dr. Daniel Haro Mendoza
