#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#==============================================================================
FASE 3: HIPER-ESPECIALIZACIÓN EN ANATOMÍA (ÓRGANOS/TARGETS)
- ResNet18 CONGELADA (Heredando visión de instrumentos)
- T-CAM y Decodificador DESCONGELADOS (Atención Espacial a Tejidos)
- 15 Clases de Anatomía Quirúrgica
#==============================================================================  
"""

import os
import sys
import time
import torch
import network
import argparse
import ivtmetrics 
import dataloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

#%% @args parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rendezvous', choices=['rendezvous'])
parser.add_argument('--version', type=int, default=100) 
parser.add_argument('--hr_output', action='store_true')
parser.add_argument('--use_ln', action='store_true')
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-e', '--test',  action='store_true')
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--data_dir', type=str, default=r'C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\cholecT50')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval')
parser.add_argument('-k', '--kfold', type=int, default=1)
parser.add_argument('--image_width', type=int, default=224)  
parser.add_argument('--image_height', type=int, default=128)  
parser.add_argument('-b', '--batch', type=int, default=4)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[5])
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01])
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--power', type=float, default=0.1)
parser.add_argument('--gpu', type=str, default="0")
FLAGS, unparsed = parser.parse_known_args()

#%% @params definitions
is_train        = FLAGS.train
val_interval    = FLAGS.val_interval
dataset_variant = FLAGS.dataset_variant
data_dir        = FLAGS.data_dir
kfold           = FLAGS.kfold
version         = FLAGS.version
batch_size      = FLAGS.batch
epochs          = FLAGS.epochs
start_epoch     = 0

# RUTAS: Volvemos a leer al Experto en Instrumentos como base
ruta_experto_instrumentos = f"./__checkpoint__/run_instrumentos_{version}/Rendezvous_ESPECIALISTA_INSTRUMENTO_resnet18_low_batch.pth"
model_dir       = f"./__checkpoint__/run_organos_{version}"
if not os.path.exists(model_dir): os.makedirs(model_dir)
ckpt_path       = os.path.join(model_dir, 'Rendezvous_ESPECIALISTA_ORGANOS.pth')
logfile         = os.path.join(model_dir, 'Rendezvous_ESPECIALISTA_ORGANOS.log')

print("Configurando red para HIPER-ESPECIALIZACIÓN de Anatomía (Órganos)...")

#%% @functions
def assign_gpu(gpu=None):  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 

# ==========================================
# CICLO DE ENTRENAMIENTO (SOLO ÓRGANOS)
# ==========================================
def train_loop(dataloader, model, loss_fn_t, optimizer_t, scheduler_t, acumulacion):
    start = time.time() 
    optimizer_t.zero_grad()
        
    for batch, (img, (_, _, y3, _)) in enumerate(dataloader):
        img, y3 = img.cuda(), y3.cuda()
        model.train()        
        
        # Extraemos solo la predicción del órgano/target (t)
        _, _, target, _ = model(img)
        _, logit_t = target
        
        loss_t = loss_fn_t(logit_t, y3.float())
        loss = loss_t / acumulacion
        loss.backward()
        
        if ((batch + 1) % acumulacion == 0) or ((batch + 1) == len(dataloader)):
            optimizer_t.step()
            optimizer_t.zero_grad()

    scheduler_t.step()
    print(f'   [INFO] Época Terminada | Error Anatomía (Órgano): {loss_t.item():.4f} | Tiempo: {(time.time() - start):.2f}s', file=open(logfile, 'a+'))   

# ==========================================
# CICLO DE EXAMEN (SOLO ÓRGANOS)
# ==========================================
def test_loop(dataloader, model, activation):
    mAPt.reset() 
    mAPt.reset_global()

    with torch.no_grad():
        for batch, (img, (_, _, y3, _)) in enumerate(dataloader):
            img, y3 = img.cuda(), y3.cuda()
            model.eval()  
            _, _, target, _ = model(img)
            
            cam_t, logit_t = target
            mAPt.update(y3.float().detach().cpu(), activation(logit_t).detach().cpu()) 
                
    mAPt.video_end()

# AUTO-GUARDADO
def weight_mgt(score, epoch):
    global benchmark
    if score > benchmark.item():
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_t.state_dict(),
            'scheduler_state_dict': scheduler_t.state_dict(),
            'benchmark': score
        }
        torch.save(checkpoint, ckpt_path)
        benchmark = torch.nn.Parameter(torch.tensor([score]), requires_grad=False)
        print(f'   [GUARDADO EXITOSO] -> Nuevo RÉCORD ANATÓMICO en Época {epoch} | Calificación T: {score:.4f}', file=open(logfile, 'a+'))  

#%% setup
assign_gpu(gpu=FLAGS.gpu)
np.seterr(divide='ignore', invalid='ignore')
torch.autograd.set_detect_anomaly(False)

dataset = dataloader.CholecT50( 
            dataset_dir=data_dir, dataset_variant=dataset_variant,
            test_fold=kfold, augmentation_list=['original', 'vflip', 'hflip', 'contrast', 'rot90']
            )

train_dataset, val_dataset, _ = dataset.build()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

#%% CONSTRUCCIÓN DEL CEREBRO (RENDEZVOUS)
model = network.Rendezvous('resnet18', hr_output=FLAGS.hr_output, use_ln=FLAGS.use_ln).cuda()

# 1. CARGAR CEREBRO DE INSTRUMENTOS (Trasplante)
print(f">>> [INGENIERÍA] Extrayendo conocimiento base de: {ruta_experto_instrumentos}")
if os.path.exists(ruta_experto_instrumentos):
    checkpoint_inst = torch.load(ruta_experto_instrumentos, map_location='cuda:0', weights_only=False)
    if 'model_state_dict' in checkpoint_inst:
        model.load_state_dict(checkpoint_inst['model_state_dict'])
    else:
        model.load_state_dict(checkpoint_inst)
    print(">>> [ÉXITO] Visión de Instrumentos inyectada correctamente.")
else:
    print(">>> [ALERTA FATAL] No se encontró el modelo de instrumentos.")
    sys.exit()

# 2. CONGELAR LOS OJOS, DESCONGELAR LA ATENCIÓN T-CAM
print(">>> [INGENIERÍA] Congelando ResNet18 y activando el módulo de Anatomía T-CAM...")
module_t = []
for name, param in model.named_parameters():
    # Si es el "tronco" visual (ResNet), lo congelamos
    if 'bottleneck' in name:
        param.requires_grad = False
    # Extraemos solo los parámetros del target para optimizar
    else:
        param.requires_grad = True
        module_t.append(param) 

benchmark = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)

#%% Loss & Optimizer
activation  = nn.Sigmoid()
loss_fn_t   = nn.BCEWithLogitsLoss() 
mAPt = ivtmetrics.Recognition(15) # 15 clases de anatomía

wp_lr = FLAGS.initial_learning_rates[0] / FLAGS.power
optimizer_t  = torch.optim.SGD(module_t, lr=wp_lr, weight_decay=FLAGS.weight_decay, momentum=0.9)
scheduler_ta = torch.optim.lr_scheduler.LinearLR(optimizer_t, start_factor=FLAGS.power, total_iters=FLAGS.warmups[0])
scheduler_tb = torch.optim.lr_scheduler.ExponentialLR(optimizer_t, gamma=FLAGS.decay_rate)
scheduler_t  = torch.optim.lr_scheduler.SequentialLR(optimizer_t, schedulers=[scheduler_ta, scheduler_tb], milestones=[FLAGS.warmups[0]+1])

#%% MOTOR DE AUTO-REANUDACIÓN
if os.path.exists(ckpt_path):
    print(f">>> [SISTEMA] Reanudando entrenamiento de ÓRGANOS desde: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    if 'epoch' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_t.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_t.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        benchmark = torch.nn.Parameter(torch.tensor([checkpoint['benchmark']]), requires_grad=False)
else:
    print(">>> [SISTEMA] Iniciando Especialista en Órganos desde cero (Época 0).")

#%% Run
header = "=================================================================================================\n"
header += f"   INICIANDO ESPECIALISTA 3: ANATOMÍA (ÓRGANOS) | ATENCIÓN ESPACIAL (T-CAM)\n"
header += "================================================================================================="
print(header, file=open(logfile, 'a+'))

acumulacion = 8 

if is_train:
    for epoch in range(start_epoch, epochs):
        try:
            lr_limpio = f"{scheduler_t.get_last_lr()[0]:.7f}"
            print(f"\n-> ENTRENANDO ÓRGANOS ÉPOCA {epoch} | LR T-CAM: {lr_limpio}", file=open(logfile, 'a+'))  
            
            train_loop(train_dataloader, model, loss_fn_t, optimizer_t, scheduler_t, acumulacion)

            if epoch % val_interval == 0:
                start = time.time()  
                print(f"   Evaluando Anatomía Quirúrgica de Época {epoch}...", file=open(logfile, 'a+'))
                test_loop(val_dataloader, model, activation)
                
                map_t_score = mAPt.compute_video_AP(ignore_null=False)['mAP']
                print(f"   [CALIFICACIONES] -> mAP_t (Órganos): {map_t_score:.4f} | ETA: {(time.time() - start):.2f}s", file=open(logfile, 'a+'))
                
                weight_mgt(map_t_score, epoch=epoch)
                
        except KeyboardInterrupt:
            print(f'\n>> [ALERTA] Entrenamiento de Órganos pausado. Archivos seguros.', file=open(logfile, 'a+'))   
            sys.exit(1)