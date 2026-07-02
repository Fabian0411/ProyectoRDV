#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
#==============================================================================
MODIFICACIÓN ESPECIAL: HIPER-ESPECIALIZACIÓN EN INSTRUMENTOS
- ResNet18 Descongelada (Fine-Tuning End-to-End)
- Batch Físico: 4 | Batch Virtual: 32 (Acumulación x8)
- Sistema de Auto-Reanudación Inteligente (Full Checkpointing)
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
parser.add_argument('--version', type=int, default=100,  help='Model version control') 
parser.add_argument('--hr_output', action='store_true')
parser.add_argument('--use_ln', action='store_true')
parser.add_argument('--decoder_layer', type=int, default=1) 
parser.add_argument('-t', '--train', action='store_true')
parser.add_argument('-e', '--test',  action='store_true')
parser.add_argument('--val_interval', type=int, default=1)
parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/Uni/TMS2/CholecT50')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval')
parser.add_argument('-k', '--kfold', type=int, default=1,  choices=[1,2,3,4,5,])
parser.add_argument('--image_width', type=int, default=224)  
parser.add_argument('--image_height', type=int, default=128)  
parser.add_argument('--augmentation_list', type=str, nargs='*', default=['original', 'vflip', 'hflip', 'contrast', 'rot90'])

parser.add_argument('-b', '--batch', type=int, default=4,  help='Batch físico conservador')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--start_epoch', type=int, default=0) # Ya no lo usaremos manualmente
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[5])
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.001])
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--decay_rate', type=float, default=0.99)
parser.add_argument('--power', type=float, default=0.1)
parser.add_argument('--gpu', type=str, default="0")
FLAGS, unparsed = parser.parse_known_args()

#%% @params definitions
is_train        = FLAGS.train
is_test         = FLAGS.test
val_interval    = FLAGS.val_interval  # <--- CORRECCIÓN APLICADA
dataset_variant = FLAGS.dataset_variant
data_dir        = FLAGS.data_dir
kfold           = FLAGS.kfold if "crossval" in dataset_variant else 0
version         = FLAGS.version
hr_output       = FLAGS.hr_output
use_ln          = FLAGS.use_ln
batch_size      = FLAGS.batch
weight_decay    = FLAGS.weight_decay
learning_rates  = FLAGS.initial_learning_rates
warmups         = FLAGS.warmups
decay_rate      = FLAGS.decay_rate
power           = FLAGS.power
epochs          = FLAGS.epochs
start_epoch     = FLAGS.start_epoch 
gpu             = FLAGS.gpu
image_height    = FLAGS.image_height
image_width     = FLAGS.image_width
decodelayer     = FLAGS.decoder_layer
addnorm         = "layer" if use_ln else "batch"
modelsize       = "high" if hr_output else "low"

# Naming
modelname       = f"Rendezvous_ESPECIALISTA_INSTRUMENTO_resnet18_{modelsize}_{addnorm}"
model_dir       = "./__checkpoint__/run_instrumentos_{}".format(version)
if not os.path.exists(model_dir): os.makedirs(model_dir)
ckpt_path       = os.path.join(model_dir, '{}.pth'.format(modelname))
logfile         = os.path.join(model_dir, '{}.log'.format(modelname))
data_augmentations = FLAGS.augmentation_list 

print("Configurando red para HIPER-ESPECIALIZACIÓN de Instrumentos...")

#%% @functions (helpers)
def assign_gpu(gpu=None):  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 

tool_weight = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]

# ==========================================
# 4. train_loop 
# ==========================================
def train_loop(dataloader, model, loss_fn_i, optimizer_i, scheduler_i, acumulacion):
    start = time.time() 
    optimizer_i.zero_grad()
        
    for batch, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
        img, y1 = img.cuda(), y1.cuda()
        model.train()        
        
        tool, _, _, _ = model(img)
        _, logit_i = tool
        
        loss_i = loss_fn_i(logit_i, y1.float())
        loss = loss_i / acumulacion
        loss.backward()
        
        if ((batch + 1) % acumulacion == 0) or ((batch + 1) == len(dataloader)):
            optimizer_i.step()
            optimizer_i.zero_grad()

    scheduler_i.step()
    print(f'   [INFO] Época Terminada | Error Visión Instrumental: {loss_i.item():.4f} | Tiempo: {(time.time() - start):.2f}s', file=open(logfile, 'a+'))    

# ==========================================
# 5. test_loop 
# ==========================================
def test_loop(dataloader, model, activation):
    mAPi.reset() 
    mAPi.reset_global()

    with torch.no_grad():
        for batch, (img, (y1, _, _, _)) in enumerate(dataloader):
            img, y1 = img.cuda(), y1.cuda()
            model.eval()  
            tool, _, _, _ = model(img)
            
            cam_i, logit_i = tool
            mAPi.update(y1.float().detach().cpu(), activation(logit_i).detach().cpu()) 
                
    mAPi.video_end()

# --- CAMBIO APLICADO: FULL CHECKPOINTING ---
def weight_mgt(score, epoch):
    global benchmark
    if score > benchmark.item():
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_i.state_dict(),
            'scheduler_state_dict': scheduler_i.state_dict(),
            'benchmark': score
        }
        torch.save(checkpoint, ckpt_path)
        benchmark = torch.nn.Parameter(torch.tensor([score]), requires_grad=False)
        print(f'   [GUARDADO EXITOSO] -> Nuevo RÉCORD VISUAL en Época {epoch} | Calificación I: {score:.4f}', file=open(logfile, 'a+'))  
        return "increased"
    else:
        return "decreased"

#%% assign device
assign_gpu(gpu=gpu)
np.seterr(divide='ignore', invalid='ignore')
torch.autograd.set_detect_anomaly(False)

#%% data loading
dataset = dataloader.CholecT50( 
            dataset_dir=data_dir, 
            dataset_variant=dataset_variant,
            test_fold=kfold, 
            augmentation_list=data_augmentations
            )

train_dataset, val_dataset, test_dataset = dataset.build()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

#%% model
model = network.Rendezvous('resnet18', hr_output=hr_output, use_ln=use_ln).cuda()
print(">>> [INGENIERÍA] Descongelando capas profunda de ResNet18...")
for param in model.parameters():
    param.requires_grad = True

benchmark = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)

#%% Loss y Métricas 
activation  = nn.Sigmoid()
loss_fn_i   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).cuda())
mAPi = ivtmetrics.Recognition(6)
mAPi.reset_global()

#%% optimizer
wp_lr = learning_rates[0] / power
module_i = []
for name, param in model.named_parameters():
    if 'cagam' not in name and 'bottleneck' not in name and 'decoder' not in name:
        module_i.append(param)

optimizer_i  = torch.optim.SGD(module_i, lr=wp_lr, weight_decay=weight_decay, momentum=0.9)
scheduler_ia = torch.optim.lr_scheduler.LinearLR(optimizer_i, start_factor=power, total_iters=warmups[0])
scheduler_ib = torch.optim.lr_scheduler.ExponentialLR(optimizer_i, gamma=decay_rate)
scheduler_i  = torch.optim.lr_scheduler.SequentialLR(optimizer_i, schedulers=[scheduler_ia, scheduler_ib], milestones=[warmups[0]+1])

#%% ==========================================
# 6. MOTOR DE AUTO-REANUDACIÓN INTELIGENTE
# ==========================================
if os.path.exists(ckpt_path):
    print(f">>> [SISTEMA] Archivo de guardado detectado: {ckpt_path}")
    
    # Dependiendo de la versión de PyTorch, map_location puede ayudar con errores de memoria
    checkpoint = torch.load(ckpt_path, map_location='cuda:0', weights_only=False)
    
    # 1. Cargamos el cerebro (si es un checkpoint viejo que solo tiene dict, lo manejamos)
    if 'epoch' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_i.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_i.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        benchmark = torch.nn.Parameter(torch.tensor([checkpoint['benchmark']]), requires_grad=False)
        print(f">>> [SISTEMA] Reanudando automáticamente desde la Época {start_epoch}")
    else:
        # Por si encuentra un .pth de los antiguos, lo carga normal desde la época 0
        model.load_state_dict(checkpoint)
        print(">>> [SISTEMA] Se cargó un modelo antiguo. Empezando en Época 0.")
else:
    print(">>> [SISTEMA] No hay guardado previo. Iniciando desde cero (Época 0).")

#%% log config
header1 = "================================================================================================="
header2 = f"   INICIANDO FINE-TUNING DE EXTREMO A EXTREMO | ESPECIALISTA: INSTRUMENTOS"
header3 = f"   Modelo: ResNet18 (DESCONGELADA) | Batch Físico: {batch_size} (Simulando 32) | Res: {image_width}x{image_height}"
header4 = "================================================================================================="
print(f"\n\n{header1}\n{header2}\n{header3}\n{header4}", file=open(logfile, 'a+'))

acumulacion = 8 

#%% run
if is_train:
    for epoch in range(start_epoch, epochs):
        try:
            lr_limpio = f"{scheduler_i.get_last_lr()[0]:.7f}"
            print(f"\n-> ENTRENANDO ESPECIALISTA ÉPOCA {epoch} | LR Fine-Tuning: {lr_limpio}", file=open(logfile, 'a+'))  
            
            train_loop(train_dataloader, model, loss_fn_i, optimizer_i, scheduler_i, acumulacion)

            if epoch % val_interval == 0:
                start = time.time()  
                mAPi.reset_global()
                print(f"   Evaluando Visión Instrumental de Época {epoch}...", file=open(logfile, 'a+'))
                test_loop(val_dataloader, model, activation)
                
                map_i_score = mAPi.compute_video_AP(ignore_null=False)['mAP']
                print(f"   [CALIFICACIONES VISUALES] -> mAP_i (Instrumentos): {map_i_score:.4f} | ETA: {(time.time() - start):.2f}s", file=open(logfile, 'a+'))
                
                behaviour = weight_mgt(map_i_score, epoch=epoch)
                
        except KeyboardInterrupt:
            print(f'\n>> [ALERTA] Experimento pausado por el usuario en Época {epoch}. Archivos seguros.', file=open(logfile, 'a+'))   
            sys.exit(1)