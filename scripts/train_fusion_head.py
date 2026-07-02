#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FASE C v4: ENTRENAMIENTO DE LA CABEZA DE FUSION
Correcciones aplicadas vs v3:
  1. Eliminado relu_final del SpatialRefinementModule (mataba señal negativa)
  2. Normalización de CAMs por video antes de entrar al módulo
  3. LR subido a 1e-4 (relu_final era el freno, no el LR)
"""

import os, sys, json, glob, torch, time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import warnings
warnings.filterwarnings("ignore")

root_dir = r"C:\ProyectoRDV"
sys.path.append(root_dir)

from src.tcn import CirugiaTCN

# ── Configuracion ─────────────────────────────────────────────────────────────
FEATURES_DIR = os.path.join(root_dir, "data", "features_1d_custom")
CAMS_DIR     = os.path.join(root_dir, "data", "cams_offline")
LABELS_DIR   = os.path.join(root_dir, "data", "CholecT50", "labels")
CKPT_TCN     = os.path.join(root_dir, "checkpoints", "TCN_FINAL.pth")
CKPT_SALIDA  = os.path.join(root_dir, "checkpoints", "SpatioTemporal_FINAL.pth")
LOG_SALIDA   = os.path.join(root_dir, "checkpoints", "fusion_training_v4.log")

NUM_EPOCHS  = 50
LR          = 1e-4   # relu_final era el freno real, no el LR
GRAD_CLIP   = 1.0
PATIENCE    = 10
BATCH_SIZE  = 32
SPLIT_VAL   = 0.2

IDX_TRIPLET, IDX_INST, IDX_VERB, IDX_TARGET = 0, 1, 7, 8

gap = nn.AdaptiveAvgPool2d((1, 1))

def cam_to_logits(cam):
    """[B, C, H, W] -> [B, C]"""
    return gap(cam).squeeze(-1).squeeze(-1)


# ── SpatialRefinementModule CORREGIDO ─────────────────────────────────────────
class SpatialRefinementModule(nn.Module):
    """
    Fusiona CAM espacial [B, C, 7, 7] con peso temporal [B, C].

    CORRECCIÓN v4: eliminado relu_final del residual.
    Las CAMs del Rendezvous tienen valores mayormente negativos
    (mean ~ -6.9, max ~ +10). El relu_final anterior convertía
    todo a cero al sumar cam_refinada + residual * cam_original,
    produciendo el estancamiento en BCE = ln(2) = 0.693.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

        self.fusion_conv = nn.Sequential(
            nn.Conv2d(num_classes * 2, num_classes, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_classes),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_classes, num_classes, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(num_classes),
        )

        # Peso residual inicializado en 1: arranca conservando la CAM original
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, cam: torch.Tensor,
                tcn_logits: torch.Tensor) -> torch.Tensor:
        """
        cam        : [B, C, 7, 7]
        tcn_logits : [B, C]  — logits crudos de la TCN (sin sigmoid)
        returns    : [B, C, 7, 7]
        """
        # Pesos temporales: sigmoid da [0, 1] con rango util por la TCN
        tcn_weights = torch.sigmoid(tcn_logits)                # [B, C]
        tcn_weights = tcn_weights.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Modular CAM con confianza temporal
        cam_modulada = cam * tcn_weights                       # [B, C, 7, 7]

        # Mezclar CAM modulada + original
        x = torch.cat([cam_modulada, cam], dim=1)              # [B, 2C, 7, 7]
        x = self.fusion_conv(x)                                # [B, C, 7, 7]

        # ── CORRECCIÓN CLAVE: sin ReLU en el residual ────────────────────────
        # ReLU aquí mataba las activaciones negativas de las CAMs,
        # produciendo salidas nulas y BCE estancado en ln(2)
        cam_refinada = x + self.residual_weight * cam          # [B, C, 7, 7]

        return cam_refinada


# ── FusionHeadTrainer ─────────────────────────────────────────────────────────
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
            # Sin ReLU final: el loss BCEWithLogitsLoss aplica sigmoid internamente
        )

        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[FusionHeadTrainer] Parametros entrenables: {n:,}")

    def forward(self, cam_i, cam_v, cam_t, tcn_i, tcn_v, tcn_t):
        cam_i_ref = self.refine_inst(cam_i, tcn_i)
        cam_v_ref = self.refine_verb(cam_v, tcn_v)
        cam_t_ref = self.refine_targ(cam_t, tcn_t)

        concat   = torch.cat([cam_i_ref, cam_v_ref, cam_t_ref], dim=1)
        cam_trip = self.triplet_head(concat)

        return cam_i_ref, cam_v_ref, cam_t_ref, cam_trip


# ── Dataset 100% en RAM ───────────────────────────────────────────────────────
class DatasetCAMs(Dataset):
    """
    Carga CAMs, logits TCN y etiquetas en RAM.
    Las CAMs se normalizan por video para centrarlas en 0
    y eliminar el sesgo de escala [-22, +10].

    CORRECCIÓN v4: str(frame_idx + 1) para alinear con JSON base-1.
    """

    def __init__(self, features_dir, cams_dir, labels_dir, tcn_model, device):
        listas = {k: [] for k in ['ci','cv','ct','ti','tv','tt',
                                   'yi','yv','yt','ytrip']}

        archivos_npy = sorted(glob.glob(os.path.join(features_dir, "*.npy")))
        print(f"  [+] Cargando {len(archivos_npy)} videos en RAM...")

        tcn_model.eval()

        for ruta_npy in archivos_npy:
            nombre = os.path.splitext(os.path.basename(ruta_npy))[0]

            ruta_ci = os.path.join(cams_dir, f"cams_inst_{nombre}.npy")
            ruta_cv = os.path.join(cams_dir, f"cams_verb_{nombre}.npy")
            ruta_ct = os.path.join(cams_dir, f"cams_targ_{nombre}.npy")
            ruta_js = os.path.join(labels_dir, f"{nombre}.json")

            if not all(os.path.exists(r) for r in [ruta_ci, ruta_cv,
                                                     ruta_ct, ruta_js]):
                print(f"      [!] {nombre}: archivos incompletos. Saltando.")
                continue

            feats  = np.load(ruta_npy).astype(np.float32)
            cams_i = np.load(ruta_ci).astype(np.float32)
            cams_v = np.load(ruta_cv).astype(np.float32)
            cams_t = np.load(ruta_ct).astype(np.float32)

            # ── CORRECCIÓN: normalizar CAMs por video ─────────────────────────
            # Elimina el sesgo de escala [-22, +10] para que fusion_conv
            # pueda aprender sin que el residual domine por magnitud
            def norm(x):
                return (x - x.mean()) / (x.std() + 1e-6)

            cams_i = norm(cams_i)
            cams_v = norm(cams_v)
            cams_t = norm(cams_t)

            with open(ruta_js, encoding="utf-8") as f:
                anotaciones = json.load(f).get("annotations", {})

            # Logits TCN pre-calculados para toda la secuencia
            with torch.no_grad():
                ft = torch.from_numpy(feats).transpose(0,1).unsqueeze(0).to(device)
                out_i, out_v, out_t, _ = tcn_model(ft)
                pesos_i = out_i.squeeze(0).transpose(0,1).cpu().numpy()
                pesos_v = out_v.squeeze(0).transpose(0,1).cpu().numpy()
                pesos_t = out_t.squeeze(0).transpose(0,1).cpu().numpy()

            T = min(feats.shape[0], cams_i.shape[0])

            for frame_idx in range(T):
                # ── CORRECCIÓN: JSON indexa desde 1, no desde 0 ───────────────
                acciones = anotaciones.get(str(frame_idx), [])

                y_i    = np.zeros(6,   dtype=np.float32)
                y_v    = np.zeros(10,  dtype=np.float32)
                y_t    = np.zeros(15,  dtype=np.float32)
                y_trip = np.zeros(100, dtype=np.float32)

                for acc in acciones:
                    if len(acc) <= max(IDX_TRIPLET, IDX_INST,
                                       IDX_VERB, IDX_TARGET):
                        continue
                    if acc[IDX_INST]    != -1: y_i[acc[IDX_INST]]       = 1.
                    if acc[IDX_VERB]    != -1: y_v[acc[IDX_VERB]]       = 1.
                    if acc[IDX_TARGET]  != -1: y_t[acc[IDX_TARGET]]     = 1.
                    if acc[IDX_TRIPLET] != -1: y_trip[acc[IDX_TRIPLET]] = 1.

                listas['ci'].append(cams_i[frame_idx])
                listas['cv'].append(cams_v[frame_idx])
                listas['ct'].append(cams_t[frame_idx])
                listas['ti'].append(pesos_i[frame_idx])
                listas['tv'].append(pesos_v[frame_idx])
                listas['tt'].append(pesos_t[frame_idx])
                listas['yi'].append(y_i)
                listas['yv'].append(y_v)
                listas['yt'].append(y_t)
                listas['ytrip'].append(y_trip)

        # Convertir a tensores una sola vez
        self.frames_i    = torch.from_numpy(np.stack(listas['ci']))
        self.frames_v    = torch.from_numpy(np.stack(listas['cv']))
        self.frames_t    = torch.from_numpy(np.stack(listas['ct']))
        self.tcn_pesos_i = torch.from_numpy(np.stack(listas['ti']))
        self.tcn_pesos_v = torch.from_numpy(np.stack(listas['tv']))
        self.tcn_pesos_t = torch.from_numpy(np.stack(listas['tt']))
        self.y_inst      = torch.from_numpy(np.stack(listas['yi']))
        self.y_verb      = torch.from_numpy(np.stack(listas['yv']))
        self.y_target    = torch.from_numpy(np.stack(listas['yt']))
        self.y_triplet   = torch.from_numpy(np.stack(listas['ytrip']))

        total = len(self.frames_i)
        pos_i = self.y_inst.sum(0)
        print(f"  [+] Dataset en RAM: {total:,} frames")
        print(f"      CAMs normalizadas: mean~0, std~1")
        print(f"      Positivos inst (top3): "
              f"{pos_i.topk(3).values.int().tolist()}")

    def __len__(self):
        return len(self.frames_i)

    def __getitem__(self, idx):
        return {
            'cam_i'    : self.frames_i[idx],
            'cam_v'    : self.frames_v[idx],
            'cam_t'    : self.frames_t[idx],
            'tcn_i'    : self.tcn_pesos_i[idx],
            'tcn_v'    : self.tcn_pesos_v[idx],
            'tcn_t'    : self.tcn_pesos_t[idx],
            'y_inst'   : self.y_inst[idx],
            'y_verb'   : self.y_verb[idx],
            'y_target' : self.y_target[idx],
            'y_triplet': self.y_triplet[idx],
        }


# ── Evaluacion ────────────────────────────────────────────────────────────────
def evaluar(modelo, loader, criterion, device):
    modelo.eval()
    acum = {'total': 0., 'i': 0., 'v': 0., 't': 0., 'ivt': 0.}
    n    = 0

    with torch.no_grad():
        for batch in loader:
            ci  = batch['cam_i'].to(device)
            cv  = batch['cam_v'].to(device)
            ct  = batch['cam_t'].to(device)
            ti  = batch['tcn_i'].to(device)
            tv  = batch['tcn_v'].to(device)
            tt  = batch['tcn_t'].to(device)
            y_i = batch['y_inst'].to(device)
            y_v = batch['y_verb'].to(device)
            y_t = batch['y_target'].to(device)
            y_trip = batch['y_triplet'].to(device)

            r_i, r_v, r_t, r_trip = modelo(ci, cv, ct, ti, tv, tt)

            l_i   = criterion(cam_to_logits(r_i),    y_i)
            l_v   = criterion(cam_to_logits(r_v),    y_v)
            l_t   = criterion(cam_to_logits(r_t),    y_t)
            l_ivt = criterion(cam_to_logits(r_trip),  y_trip)

            bs = ci.shape[0]
            acum['i']     += l_i.item()   * bs
            acum['v']     += l_v.item()   * bs
            acum['t']     += l_t.item()   * bs
            acum['ivt']   += l_ivt.item() * bs
            acum['total'] += (l_i + l_v + l_t + l_ivt * 2.).item() * bs
            n += bs

    return {k: v / max(n, 1) for k, v in acum.items()}


# ── Setup ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"-> Dispositivo: {device}\n")

print("-> Cargando TCN para pre-calculo de pesos...")
tcn_model = CirugiaTCN().to(device)
raw = torch.load(CKPT_TCN, map_location=device, weights_only=False)
tcn_model.load_state_dict(raw.get('model_state_dict', raw))

print("-> Preparando dataset (carga en RAM, ~1 min)...")
dataset_completo = DatasetCAMs(FEATURES_DIR, CAMS_DIR, LABELS_DIR,
                                tcn_model, device)
del tcn_model
torch.cuda.empty_cache()

n_val   = int(len(dataset_completo) * SPLIT_VAL)
n_train = len(dataset_completo) - n_val

ds_train, ds_val = random_split(
    dataset_completo, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0, pin_memory=False)
loader_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0, pin_memory=False)

print("\n-> Construyendo FusionHeadTrainer v4...")
modelo    = FusionHeadTrainer().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(modelo.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=False
)

# ── Reanudar ──────────────────────────────────────────────────────────────────
epoca_inicio      = 0
mejor_loss        = float('inf')
epocas_sin_mejora = 0

if os.path.exists(CKPT_SALIDA):
    prev = torch.load(CKPT_SALIDA, map_location=device, weights_only=False)
    if 'fusion_head_state' in prev:
        modelo.load_state_dict(prev['fusion_head_state'])
        optimizer.load_state_dict(prev['optimizer'])
        epoca_inicio      = prev.get('epoch', 0)
        mejor_loss        = prev.get('mejor_loss', float('inf'))
        epocas_sin_mejora = prev.get('epocas_sin_mejora', 0)
        print(f"[+] Reanudando desde epoca {epoca_inicio + 1}, "
              f"mejor loss: {mejor_loss:.4f}")
    else:
        print("[+] Checkpoint incompatible con v4. Entrenando desde cero.")
else:
    print("[+] Entrenamiento desde cero.")

# ── Log ───────────────────────────────────────────────────────────────────────
modo_log = "a" if epoca_inicio > 0 else "w"
with open(LOG_SALIDA, modo_log, encoding="utf-8") as f:
    if epoca_inicio == 0:
        f.write("Epoca,Train_Total,Train_I,Train_V,Train_T,Train_IVT,"
                "Val_Total,Val_I,Val_V,Val_T,Val_IVT,LR,Tiempo_s\n")

n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
print("\n" + "=" * 57)
print("   FASE C v4: ENTRENAMIENTO SIN IMAGENES (CORREGIDO)")
print("=" * 57)
print(f"   Parametros entrenables : {n_params:,}")
print(f"   Batch size             : {BATCH_SIZE}")
print(f"   LR inicial             : {LR:.1e}")
print(f"   Items train / val      : {n_train:,} / {n_val:,}")
print(f"   Correcciones           : relu_final removido, CAMs normalizadas,")
print(f"                           JSON offset +1 corregido")
print(f"   Dispositivo            : {device}")
print("=" * 57 + "\n")

# ── Entrenamiento ─────────────────────────────────────────────────────────────
params_lista = list(modelo.parameters())

for epoch in range(epoca_inicio, NUM_EPOCHS):
    t_inicio = time.time()
    modelo.train()

    acum = {'total': 0., 'i': 0., 'v': 0., 't': 0., 'ivt': 0.}
    n    = 0

    for batch in loader_train:
        ci  = batch['cam_i'].to(device)
        cv  = batch['cam_v'].to(device)
        ct  = batch['cam_t'].to(device)
        ti  = batch['tcn_i'].to(device)
        tv  = batch['tcn_v'].to(device)
        tt  = batch['tcn_t'].to(device)
        y_i = batch['y_inst'].to(device)
        y_v = batch['y_verb'].to(device)
        y_t = batch['y_target'].to(device)
        y_trip = batch['y_triplet'].to(device)

        optimizer.zero_grad()

        r_i, r_v, r_t, r_trip = modelo(ci, cv, ct, ti, tv, tt)

        l_i   = criterion(cam_to_logits(r_i),    y_i)
        l_v   = criterion(cam_to_logits(r_v),    y_v)
        l_t   = criterion(cam_to_logits(r_t),    y_t)
        l_ivt = criterion(cam_to_logits(r_trip),  y_trip)

        loss = l_i + l_v + l_t + l_ivt * 2.0
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params_lista, GRAD_CLIP)
        optimizer.step()

        bs = ci.shape[0]
        acum['i']     += l_i.item()   * bs
        acum['v']     += l_v.item()   * bs
        acum['t']     += l_t.item()   * bs
        acum['ivt']   += l_ivt.item() * bs
        acum['total'] += loss.item()  * bs
        n += bs

    metrics_train = {k: v / max(n, 1) for k, v in acum.items()}
    metrics_val   = evaluar(modelo, loader_val, criterion, device)
    scheduler.step(metrics_val['total'])

    lr_actual    = optimizer.param_groups[0]['lr']
    nuevo_record = metrics_val['total'] < mejor_loss
    t_epoca      = int(time.time() - t_inicio)
    mins, segs   = divmod(t_epoca, 60)

    print(f"[Epoca {epoch+1:03d}/{NUM_EPOCHS}]  Tiempo: {mins}m {segs:02d}s")
    print(f"  Train -> Total: {metrics_train['total']:.4f}"
          f"  (I:{metrics_train['i']:.3f}"
          f"  V:{metrics_train['v']:.3f}"
          f"  T:{metrics_train['t']:.3f}"
          f"  IVT:{metrics_train['ivt']:.3f})")
    print(f"  Val   -> Total: {metrics_val['total']:.4f}"
          f"  (I:{metrics_val['i']:.3f}"
          f"  V:{metrics_val['v']:.3f}"
          f"  T:{metrics_val['t']:.3f}"
          f"  IVT:{metrics_val['ivt']:.3f})")
    print(f"  LR: {lr_actual:.2e}", end="")

    if nuevo_record:
        mejor_loss        = metrics_val['total']
        epocas_sin_mejora = 0
        print("  NUEVO RECORD -- Guardando...")
        torch.save({
            'epoch'            : epoch + 1,
            'mejor_loss'       : mejor_loss,
            'epocas_sin_mejora': 0,
            'fusion_head_state': modelo.state_dict(),
            'optimizer'        : optimizer.state_dict(),
        }, CKPT_SALIDA)
    else:
        epocas_sin_mejora += 1
        print(f"  Sin mejora: {epocas_sin_mejora}/{PATIENCE}")

    print()

    with open(LOG_SALIDA, "a", encoding="utf-8") as f:
        f.write(f"{epoch+1},"
                f"{metrics_train['total']:.4f},{metrics_train['i']:.4f},"
                f"{metrics_train['v']:.4f},{metrics_train['t']:.4f},"
                f"{metrics_train['ivt']:.4f},"
                f"{metrics_val['total']:.4f},{metrics_val['i']:.4f},"
                f"{metrics_val['v']:.4f},{metrics_val['t']:.4f},"
                f"{metrics_val['ivt']:.4f},"
                f"{lr_actual:.2e},{t_epoca}\n")

    if epocas_sin_mejora >= PATIENCE:
        print("-" * 57)
        print(f"[EARLY STOPPING] Sin mejora por {PATIENCE} epocas.")
        print(f"  Mejor val_loss : {mejor_loss:.4f}")
        break

print("=" * 57)
print("  ENTRENAMIENTO FINALIZADO")
print(f"  Mejor val_loss : {mejor_loss:.4f}")
print(f"  Checkpoint     : {CKPT_SALIDA}")
print(f"  Log            : {LOG_SALIDA}")
print("=" * 57)