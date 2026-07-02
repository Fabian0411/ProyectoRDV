#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#==============================================================================
FASE 4: LA GRAN FUSIÓN (END-TO-END FINE TUNING)
- Se inyectan los 3 especialistas (Instrumentos, Verbos, Órganos) en un solo modelo.
- DESCONGELAMIENTO TOTAL (Todas las capas son entrenables).
- Protección activa: Learning Rate Microscópico (1e-5) y Gradient Clipping.
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
parser.add_argument('--data_dir', type=str, default=r'C:\Users\Nan\OneDrive\Documentos\Servicio Social\LSTM\CholecT50')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval')
parser.add_argument('-k', '--kfold', type=int, default=1)
parser.add_argument('--image_width', type=int, default=224)  
parser.add_argument('--image_height', type=int, default=128)  
parser.add_argument('-b', '--batch', type=int, default=4)
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[3])
# REGLA DE ORO 1: Learning Rate Microscópico para no destruir el conocimiento base
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[1e-5]) 
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

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS (Versión Robusta)
# ==========================================
base_ckpt_dir = os.path.join(".", "__checkpoint__")
ruta_organos = os.path.join(base_ckpt_dir, f"run_organos_{version}", "Rendezvous_ESPECIALISTA_ORGANOS.pth")
ruta_verbos  = os.path.join(base_ckpt_dir, f"run_verbos_{version}", "Rendezvous_ESPECIALISTA_VERBOS.pth")

model_dir    = os.path.join(base_ckpt_dir, f"run_fusion_{version}")
if not os.path.exists(model_dir): os.makedirs(model_dir)
ckpt_path       = os.path.join(model_dir, 'Rendezvous_FUSION_FINAL.pth')
logfile         = os.path.join(model_dir, 'Rendezvous_FUSION_FINAL.log')

def assign_gpu(gpu=None):  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 

# ==========================================
# CICLO DE ENTRENAMIENTO SIMULTÁNEO
# ==========================================
def train_loop(dataloader, model, loss_fns, optimizer, scheduler, acumulacion):
    start = time.time() 
    optimizer.zero_grad()
    loss_fn_i, loss_fn_v, loss_fn_t = loss_fns
        
    for batch, (img, (y1, y2, y3, _)) in enumerate(dataloader):
        img, y1, y2, y3 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda()
        model.train()        
        
        tool, verb, target, _ = model(img)
        _, logit_i = tool
        _, logit_v = verb
        _, logit_t = target
        
        loss_i = loss_fn_i(logit_i, y1.float())
        loss_v = loss_fn_v(logit_v, y2.float())
        loss_t = loss_fn_t(logit_t, y3.float())
        
        loss_total = (loss_i + loss_v + loss_t) / acumulacion
        loss_total.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        
        if ((batch + 1) % acumulacion == 0) or ((batch + 1) == len(dataloader)):
            optimizer.step()
            optimizer.zero_grad()

    scheduler.step()
    print(f'   [INFO] Época Terminada | Error Fusión (I+V+T): {loss_total.item()*acumulacion:.4f} | Tiempo: {(time.time() - start):.2f}s', file=open(logfile, 'a+'))   

# ==========================================
# CICLO DE EXAMEN (EVALUACIÓN DE TRIPLETE COMPLETO)
# ==========================================
def test_loop(dataloader, model, activation):
    mAPi.reset(); mAPi.reset_global()
    mAPv.reset(); mAPv.reset_global()
    mAPt.reset(); mAPt.reset_global()
    mAPivt.reset(); mAPivt.reset_global() # Evaluador de 100 clases activado

    with torch.no_grad():
        for batch, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
            img = img.cuda()
            model.eval()  
            
            # ¡LA MAGIA! Extraemos el 4to valor directamente (logit_ivt)
            tool, verb, target, logit_ivt = model(img)
            
            _, logit_i = tool
            _, logit_v = verb
            _, logit_t = target
            
            prob_i = activation(logit_i).detach().cpu()
            prob_v = activation(logit_v).detach().cpu()
            prob_t = activation(logit_t).detach().cpu()
            prob_ivt = activation(logit_ivt).detach().cpu() # Triplete procesado por la red
            
            # Evaluamos las partes individuales
            mAPi.update(y1.float().detach().cpu(), prob_i) 
            mAPv.update(y2.float().detach().cpu(), prob_v) 
            mAPt.update(y3.float().detach().cpu(), prob_t) 
            
            # Calificamos nuestra predicción contra la boleta oficial y4
            mAPivt.update(y4.float().detach().cpu(), prob_ivt) 
            
    mAPi.video_end(); mAPv.video_end(); mAPt.video_end(); mAPivt.video_end()

# AUTO-GUARDADO BASADO EN EL PROMEDIO GLOBAL
def weight_mgt(score, epoch):
    global benchmark
    if score > benchmark.item():
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'benchmark': score
        }
        torch.save(checkpoint, ckpt_path)
        benchmark = torch.nn.Parameter(torch.tensor([score]), requires_grad=False)
        print(f'   [GUARDADO EXITOSO] -> Nuevo RÉCORD DE FUSIÓN en Época {epoch} | Calificación Global: {score:.4f}', file=open(logfile, 'a+'))  

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

#%% CONSTRUCCIÓN DEL FRANKENSTEIN (RENDEZVOUS)
model = network.Rendezvous('resnet18', hr_output=FLAGS.hr_output, use_ln=FLAGS.use_ln).cuda()

print(">>> [INGENIERÍA] Iniciando Protocolo de Fusión de Especialistas...")

if os.path.exists(ruta_organos):
    ckpt_organos = torch.load(ruta_organos, map_location='cuda:0', weights_only=False)
    state_dict_organos = ckpt_organos.get('model_state_dict', ckpt_organos)
    model.load_state_dict(state_dict_organos, strict=False)
    print(">>> [ÉXITO] Base de Instrumentos y T-CAM (Órganos) cargada.")
else:
    print(f">>> [ALERTA FATAL] No se encontró el modelo de órganos en: {ruta_organos}")
    sys.exit()

if os.path.exists(ruta_verbos):
    ckpt_verbos = torch.load(ruta_verbos, map_location='cuda:0', weights_only=False)
    state_dict_verbos = ckpt_verbos.get('model_state_dict', ckpt_verbos)
    
    dict_modelo_actual = model.state_dict()
    pesos_verbos = {k: v for k, v in state_dict_verbos.items() if 'action' in k or 'cagam' in k or 'verb' in k}
    dict_modelo_actual.update(pesos_verbos)
    model.load_state_dict(dict_modelo_actual)
    print(">>> [ÉXITO] Módulo CAGAM (Verbos) inyectado exitosamente. Cerebro completo.")
else:
    print(f">>> [ALERTA FATAL] No se encontró el modelo de verbos en: {ruta_verbos}")
    sys.exit()

print(">>> [INGENIERÍA] Descongelando el 100% de la red neuronal...")
for param in model.parameters():
    param.requires_grad = True

benchmark = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)

#%% Loss & Optimizer
activation  = nn.Sigmoid()
loss_fns    = (nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss(), nn.BCEWithLogitsLoss())
mAPi = ivtmetrics.Recognition(6)
mAPv = ivtmetrics.Recognition(10)
mAPt = ivtmetrics.Recognition(15)
mAPivt = ivtmetrics.Recognition(100) # Inicializador del métrico

wp_lr = FLAGS.initial_learning_rates[0] / FLAGS.power
optimizer  = torch.optim.SGD(model.parameters(), lr=wp_lr, weight_decay=FLAGS.weight_decay, momentum=0.9)
scheduler_a = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=FLAGS.power, total_iters=FLAGS.warmups[0])
scheduler_b = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=FLAGS.decay_rate)
scheduler  = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_a, scheduler_b], milestones=[FLAGS.warmups[0]+1])

#%% MOTOR DE AUTO-REANUDACIÓN
if os.path.exists(ckpt_path):
    print(f">>> [SISTEMA] Reanudando FUSIÓN desde: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    if 'epoch' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        benchmark = torch.nn.Parameter(torch.tensor([checkpoint['benchmark']]), requires_grad=False)
else:
    print(">>> [SISTEMA] Iniciando Fusión End-to-End desde cero (Época 0).")

#%% Run
header = "=================================================================================================\n"
header += f"   INICIANDO FASE 4: FUSIÓN DE ESPECIALISTAS (END-TO-END) | RENDEZVOUS COMPLETO\n"
header += "================================================================================================="
print(header, file=open(logfile, 'a+'))

acumulacion = 8 

if is_train:
    for epoch in range(start_epoch, epochs):
        try:
            lr_limpio = f"{scheduler.get_last_lr()[0]:.7f}"
            print(f"\n-> ENTRENANDO FUSIÓN ÉPOCA {epoch} | LR Global: {lr_limpio}", file=open(logfile, 'a+'))  
            
            train_loop(train_dataloader, model, loss_fns, optimizer, scheduler, acumulacion)

            if epoch % val_interval == 0:
                start = time.time()  
                print(f"   Evaluando Triplete Quirúrgico de Época {epoch}...", file=open(logfile, 'a+'))
                test_loop(val_dataloader, model, activation)
                
                map_i_score = mAPi.compute_video_AP(ignore_null=False)['mAP']
                map_v_score = mAPv.compute_video_AP(ignore_null=False)['mAP']
                map_t_score = mAPt.compute_video_AP(ignore_null=False)['mAP']
                map_ivt_score = mAPivt.compute_video_AP(ignore_null=False)['mAP'] 
                
                score_global = (map_i_score + map_v_score + map_t_score) / 3.0
                
                res = f"   [DESGLOSE] -> Inst: {map_i_score:.4f} | Verbo: {map_v_score:.4f} | Órgano: {map_t_score:.4f}\n"
                res += f"   [PROMEDIO] -> {score_global:.4f} | ETA: {(time.time() - start):.2f}s\n"
                res += f"   [MÉTRICA GLOBAL AP_IVT (TRIPLETE COMPLETO)] -> {map_ivt_score:.4f}"
                print(res, file=open(logfile, 'a+'))
                
                weight_mgt(score_global, epoch=epoch)
                
        except KeyboardInterrupt:
            print(f'\n>> [ALERTA] Entrenamiento de Fusión pausado. Archivos seguros.', file=open(logfile, 'a+'))   
            sys.exit(1)