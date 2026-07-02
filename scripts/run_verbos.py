#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#==============================================================================
FASE 2: HIPER-ESPECIALIZACIÓN EN ACCIONES (VERBOS)
- ResNet18 CONGELADA (Heredando el conocimiento del Especialista 1)
- CAGAM y Decodificador DESCONGELADOS (Atención Espacial Activa)
- 10 Clases de Verbos Quirúrgicos
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
parser.add_argument('-b', '--batch', type=int, default=4, help='Batch físico conservador')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[5])
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01]) # LR un poco más alto porque es red nueva
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

# RUTAS IMPORTANTES
ruta_experto_instrumentos = f"./__checkpoint__/run_instrumentos_{version}/Rendezvous_ESPECIALISTA_INSTRUMENTO_resnet18_low_batch.pth"
model_dir       = f"./__checkpoint__/run_verbos_{version}"
if not os.path.exists(model_dir): os.makedirs(model_dir)
ckpt_path       = os.path.join(model_dir, 'Rendezvous_ESPECIALISTA_VERBOS.pth')
logfile         = os.path.join(model_dir, 'Rendezvous_ESPECIALISTA_VERBOS.log')

print("Configurando red para HIPER-ESPECIALIZACIÓN de Acciones (Verbos)...")

#%% @functions
def assign_gpu(gpu=None):  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 

# ==========================================
# CICLO DE ENTRENAMIENTO (SOLO VERBOS)
# ==========================================
def train_loop(dataloader, model, loss_fn_v, optimizer_v, scheduler_v, acumulacion):
    start = time.time() 
    optimizer_v.zero_grad()
        
    for batch, (img, (_, y2, _, _)) in enumerate(dataloader):
        img, y2 = img.cuda(), y2.cuda()
        model.train()        
        
        # Extraemos solo la predicción del verbo (v)
        _, verb, _, _ = model(img)
        _, logit_v = verb
        
        loss_v = loss_fn_v(logit_v, y2.float())
        loss = loss_v / acumulacion
        loss.backward()
        
        if ((batch + 1) % acumulacion == 0) or ((batch + 1) == len(dataloader)):
            optimizer_v.step()
            optimizer_v.zero_grad()

    scheduler_v.step()
    print(f'   [INFO] Época Terminada | Error Acción (Verbo): {loss_v.item():.4f} | Tiempo: {(time.time() - start):.2f}s', file=open(logfile, 'a+'))    

# ==========================================
# CICLO DE EXAMEN (SOLO VERBOS)
# ==========================================
def test_loop(dataloader, model, activation):
    mAPv.reset() 
    mAPv.reset_global()

    with torch.no_grad():
        for batch, (img, (_, y2, _, _)) in enumerate(dataloader):
            img, y2 = img.cuda(), y2.cuda()
            model.eval()  
            _, verb, _, _ = model(img)
            
            cam_v, logit_v = verb
            mAPv.update(y2.float().detach().cpu(), activation(logit_v).detach().cpu()) 
                
    mAPv.video_end()

# AUTO-GUARDADO INTELIGENTE
def weight_mgt(score, epoch):
    global benchmark
    if score > benchmark.item():
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_v.state_dict(),
            'scheduler_state_dict': scheduler_v.state_dict(),
            'benchmark': score
        }
        torch.save(checkpoint, ckpt_path)
        benchmark = torch.nn.Parameter(torch.tensor([score]), requires_grad=False)
        print(f'   [GUARDADO EXITOSO] -> Nuevo RÉCORD DE ACCIÓN en Época {epoch} | Calificación V: {score:.4f}', file=open(logfile, 'a+'))  

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
print(f">>> [INGENIERÍA] Extrayendo conocimiento base del archivo: {ruta_experto_instrumentos}")
if os.path.exists(ruta_experto_instrumentos):
    checkpoint_inst = torch.load(ruta_experto_instrumentos, map_location='cuda:0', weights_only=False)
    if 'model_state_dict' in checkpoint_inst:
        model.load_state_dict(checkpoint_inst['model_state_dict'])
    else:
        model.load_state_dict(checkpoint_inst)
    print(">>> [ÉXITO] Visión de Instrumentos inyectada correctamente.")
else:
    print(">>> [ALERTA FATAL] No se encontró el modelo de instrumentos. Revisa la ruta.")
    sys.exit()

# 2. CONGELAR LOS OJOS, DESCONGELAR LA ATENCIÓN
print(">>> [INGENIERÍA] Congelando ResNet18 y activando el módulo de Atención CAGAM...")
module_v = []
for name, param in model.named_parameters():
    # Si es el "tronco" visual (ResNet), lo congelamos
    if 'bottleneck' in name:
        param.requires_grad = False
    # Si es el sistema de atención o el decodificador, lo liberamos
    else:
        param.requires_grad = True
        module_v.append(param) # Solo mandamos estas piezas al optimizador

benchmark = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)

#%% Loss & Optimizer
activation  = nn.Sigmoid()
loss_fn_v   = nn.BCEWithLogitsLoss() # Pérdida estándar para los 10 verbos
mAPv = ivtmetrics.Recognition(10) # 10 clases de acciones

wp_lr = FLAGS.initial_learning_rates[0] / FLAGS.power
optimizer_v  = torch.optim.SGD(module_v, lr=wp_lr, weight_decay=FLAGS.weight_decay, momentum=0.9)
scheduler_va = torch.optim.lr_scheduler.LinearLR(optimizer_v, start_factor=FLAGS.power, total_iters=FLAGS.warmups[0])
scheduler_vb = torch.optim.lr_scheduler.ExponentialLR(optimizer_v, gamma=FLAGS.decay_rate)
scheduler_v  = torch.optim.lr_scheduler.SequentialLR(optimizer_v, schedulers=[scheduler_va, scheduler_vb], milestones=[FLAGS.warmups[0]+1])

#%% MOTOR DE AUTO-REANUDACIÓN
if os.path.exists(ckpt_path):
    print(f">>> [SISTEMA] Reanudando entrenamiento de VERBOS desde: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    if 'epoch' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_v.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_v.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        benchmark = torch.nn.Parameter(torch.tensor([checkpoint['benchmark']]), requires_grad=False)
else:
    print(">>> [SISTEMA] Iniciando Especialista en Verbos desde cero (Época 0).")

#%% Run
header = "=================================================================================================\n"
header += f"   INICIANDO ESPECIALISTA 2: ACCIONES (VERBOS) | ATENCIÓN ESPACIAL (CAGAM)\n"
header += "================================================================================================="
print(header, file=open(logfile, 'a+'))

acumulacion = 8 

if is_train:
    for epoch in range(start_epoch, epochs):
        try:
            lr_limpio = f"{scheduler_v.get_last_lr()[0]:.7f}"
            print(f"\n-> ENTRENANDO VERBOS ÉPOCA {epoch} | LR Atención: {lr_limpio}", file=open(logfile, 'a+'))  
            
            train_loop(train_dataloader, model, loss_fn_v, optimizer_v, scheduler_v, acumulacion)

            if epoch % val_interval == 0:
                start = time.time()  
                print(f"   Evaluando Acciones Quirúrgicas de Época {epoch}...", file=open(logfile, 'a+'))
                test_loop(val_dataloader, model, activation)
                
                map_v_score = mAPv.compute_video_AP(ignore_null=False)['mAP']
                print(f"   [CALIFICACIONES] -> mAP_v (Acciones): {map_v_score:.4f} | ETA: {(time.time() - start):.2f}s", file=open(logfile, 'a+'))
                
                weight_mgt(map_v_score, epoch=epoch)
                
        except KeyboardInterrupt:
            print(f'\n>> [ALERTA] Entrenamiento de Verbos pausado. Archivos seguros.', file=open(logfile, 'a+'))   
            sys.exit(1)