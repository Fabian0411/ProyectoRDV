#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CODE RELEASE TO SUPPORT RESEARCH.
COMMERCIAL USE IS NOT PERMITTED.
#==============================================================================
An implementation based on:
***
    C.I. Nwoye, T. Yu, C. Gonzalez, B. Seeliger, P. Mascagni, D. Mutter, J. Marescaux, N. Padoy. 
    Rendezvous: Attention Mechanisms for the Recognition of Surgical Action Triplets in Endoscopic Videos. 
    Medical Image Analysis, 78 (2022) 102433.
*** Created on Thu Oct 21 15:38:36 2021
#==============================================================================  
Copyright 2021 The Research Group CAMMA Authors All Rights Reserved.
(c) Research Group CAMMA, University of Strasbourg, France
@ Laboratory: CAMMA - ICube
@ Author: Nwoye Chinedu Innocent
@ Website: http://camma.u-strasbg.fr
#==============================================================================
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
#==============================================================================
"""

#%% import libraries
import os
import sys
import time
import torch
import random
import network
import argparse
import platform
import ivtmetrics 
import dataloader
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

#%% @args parsing
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rendezvous', choices=['rendezvous'], help='Model name?')
parser.add_argument('--version', type=int, default=0,  help='Model version control (for keeping several versions)') 
parser.add_argument('--hr_output', action='store_true', help='Whether to use higher resolution output (32x56) or not (8x14). Default: False')
parser.add_argument('--use_ln', action='store_true', help='Whether to use layer norm or batch norm in AddNorm() function. Default: False')
parser.add_argument('--decoder_layer', type=int, default=8, help='Number of MHMA layers ') 
parser.add_argument('-t', '--train', action='store_true', help='to train.')
parser.add_argument('-e', '--test',  action='store_true', help='to test')
parser.add_argument('--val_interval', type=int, default=1,  help='(for hp tuning). Epoch interval to evaluate on validation data.')
parser.add_argument('--data_dir', type=str, default='/content/drive/MyDrive/Uni/TMS2/CholecT50', help='path to dataset?')
parser.add_argument('--dataset_variant', type=str, default='cholect45-crossval', choices=['cholect50', 'cholect45', 'cholect50-challenge', 'cholect50-crossval', 'cholect45-crossval'], help='Variant of the dataset to use')
parser.add_argument('-k', '--kfold', type=int, default=1,  choices=[1,2,3,4,5,], help='The test split in k-fold cross-validation')
parser.add_argument('--image_width', type=int, default=448, help='Image width ')  
parser.add_argument('--image_height', type=int, default=256, help='Image height ')  
parser.add_argument('--image_channel', type=int, default=3, help='Image channels ')  
parser.add_argument('--num_tool_classes', type=int, default=6, help='Number of tool categories')
parser.add_argument('--num_verb_classes', type=int, default=10, help='Number of verb categories')
parser.add_argument('--num_target_classes', type=int, default=15, help='Number of target categories')
parser.add_argument('--num_triplet_classes', type=int, default=100, help='Number of triplet categories')
parser.add_argument('--augmentation_list', type=str, nargs='*', default=['original', 'vflip', 'hflip', 'contrast', 'rot90'], help='List augumentation styles.')
parser.add_argument('-b', '--batch', type=int, default=32,  help='The size of sample training batch')
parser.add_argument('--epochs', type=int, default=100,  help='How many training epochs?')
parser.add_argument('--start_epoch', type=int, default=0,  help='Epoch to resume training from')
parser.add_argument('-w', '--warmups', type=int, nargs='+', default=[9,18,58], help='List warmup epochs for tool, verb-target, triplet respectively')
parser.add_argument('-l', '--initial_learning_rates', type=float, nargs='+', default=[0.01, 0.01, 0.01], help='List learning rates for tool, verb-target, triplet respectively')
parser.add_argument('--weight_decay', type=float, default=1e-5,  help='L2 regularization weight decay constant')
parser.add_argument('--decay_steps', type=int, default=10,  help='Step to exponentially decay')
parser.add_argument('--decay_rate', type=float, default=0.99,  help='Learning rates weight decay rate')
parser.add_argument('--momentum', type=float, default=0.95,  help="Optimizer's momentum")
parser.add_argument('--power', type=float, default=0.1,  help='Learning rates weight decay power')
parser.add_argument('--pretrain_dir', type=str, default='', help='path to pretrain_weight?')
parser.add_argument('--test_ckpt', type=str, default=None, help='path to model weight for testing')
parser.add_argument('--gpu', type=str, default="0",  help='The gpu device to use.')
FLAGS, unparsed = parser.parse_known_args()

#%% @params definitions
is_train        = FLAGS.train
is_test         = FLAGS.test
dataset_variant = FLAGS.dataset_variant
data_dir        = FLAGS.data_dir
kfold           = FLAGS.kfold if "crossval" in dataset_variant else 0
version         = FLAGS.version
hr_output       = FLAGS.hr_output
use_ln          = FLAGS.use_ln
batch_size      = FLAGS.batch
pretrain_dir    = FLAGS.pretrain_dir
test_ckpt       = FLAGS.test_ckpt
weight_decay    = FLAGS.weight_decay
learning_rates  = FLAGS.initial_learning_rates
warmups         = FLAGS.warmups
decay_steps     = FLAGS.decay_steps
decay_rate      = FLAGS.decay_rate
power           = FLAGS.power
momentum        = FLAGS.momentum
epochs          = FLAGS.epochs
start_epoch     = FLAGS.start_epoch 
gpu             = FLAGS.gpu
image_height    = FLAGS.image_height
image_width     = FLAGS.image_width
image_channel   = FLAGS.image_channel
num_triplet     = FLAGS.num_triplet_classes
num_tool        = FLAGS.num_tool_classes
num_verb        = FLAGS.num_verb_classes
num_target      = FLAGS.num_target_classes
val_interval    = FLAGS.epochs-1 if FLAGS.val_interval==-1 else FLAGS.val_interval
set_chlg_eval   = True if "challenge" in dataset_variant else False 
gpu             = ",".join(str(FLAGS.gpu).split(","))
decodelayer     = FLAGS.decoder_layer
addnorm         = "layer" if use_ln else "batch"
modelsize       = "high" if hr_output else "low"
FLAGS.multigpu  = len(gpu) > 1  
mheaders        = ["","l", "cholect", "k"]
margs           = [FLAGS.model, decodelayer, dataset_variant, kfold]
wheaders        = ["norm", "res"]
wargs           = [addnorm, modelsize]
modelname       = "_".join(["{}{}".format(x,y) for x,y in zip(mheaders, margs) if len(str(y))])+"_"+\
                  "_".join(["{}{}".format(x,y) for x,y in zip(wargs, wheaders) if len(str(x))])
model_dir       = "./__checkpoint__/run_{}".format(version)
if not os.path.exists(model_dir): os.makedirs(model_dir)
resume_ckpt     = None
ckpt_path       = os.path.join(model_dir, '{}.pth'.format(modelname))
logfile         = os.path.join(model_dir, '{}.log'.format(modelname))
data_augmentations      = FLAGS.augmentation_list 
iterable_augmentations  = []
print("Configuring network ...")

#%% @functions (helpers)
def assign_gpu(gpu=None):  
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu) 
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1' 

def get_weight_balancing(case='cholect50'):
    switcher = {
        'cholect50': {
            'tool'  :   [0.08084519, 0.81435289, 0.10459284, 2.55976864, 1.630372490, 1.29528455],
            'verb'  :   [0.31956735, 0.07252306, 0.08111481, 0.81137309, 1.302895320, 2.12264151, 1.54109589, 8.86363636, 12.13692946, 0.40462028],
            'target':   [0.06246232, 1.00000000, 0.34266478, 0.84750219, 14.80102041, 8.73795181, 1.52845100, 5.74455446, 0.285756500, 12.72368421, 0.6250808,  3.85771277, 6.95683453, 0.84923888, 0.40130032]
        },
        'cholect45-crossval': {
            1: {
                'tool':     [0.08165644, 0.91226868, 0.10674758, 2.85418156, 1.60554885, 1.10640067],
                'verb':     [0.37870137, 0.06836869, 0.07931255, 0.84780024, 1.21880342, 2.52836879, 1.30765704, 6.88888889, 17.07784431, 0.45241117],
                'target':   [0.07149629, 1.0, 0.41013597, 0.90458015, 13.06299213, 12.06545455, 1.5213205, 5.04255319, 0.35808332, 45.45205479, 0.67493897, 7.04458599, 9.14049587, 0.97330595, 0.52633249]
                }
        }
    }
    return switcher.get(case, switcher['cholect45-crossval'])
     
def train_loop(dataloader, model, activation, loss_fn_i, loss_fn_v, loss_fn_t, loss_fn_ivt, optimizers, scheduler, epoch):
    start = time.time() 
    
    # --- ACUMULACIÓN DE GRADIENTES ---
    acumulacion = 8 # Simula Batch 32 con un Batch real de 4 (8 x 4 = 32)
    
    for param in model.parameters():
        param.grad = None
        
    for batch, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
        img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()        
        model.train()        
        
        tool, verb, target, triplet = model(img)
        cam_i, logit_i  = tool
        cam_v, logit_v  = verb
        cam_t, logit_t  = target
        logit_ivt       = triplet                
        
        loss_i          = loss_fn_i(logit_i, y1.float())
        loss_v          = loss_fn_v(logit_v, y2.float())
        loss_t          = loss_fn_t(logit_t, y3.float())
        loss_ivt        = loss_fn_ivt(logit_ivt, y4.float())  
        
        loss            = (loss_i) + (loss_v) + (loss_t) + loss_ivt 
        loss            = loss / acumulacion # Trampa matemática para promediar
        
        loss.backward()
        
        # Actualiza solo cada 8 ciclos
        if ((batch + 1) % acumulacion == 0) or ((batch + 1) == len(dataloader)):
            for opt in optimizers:
                opt.step()
            for param in model.parameters():
                param.grad = None

    for sch in scheduler:
        sch.step()
        
    print(f'   Terminado | Error (Loss) -> Instr: {loss_i.item():.4f} | Verbos: {loss_v.item():.4f} | Órganos: {loss_t.item():.4f} | Global: {loss_ivt.item():.4f} | Tiempo: {(time.time() - start):.2f}s', file=open(logfile, 'a+'))    

def test_loop(dataloader, model, activation, final_eval=False):
    mAP.reset()  
    if not set_chlg_eval:
        mAPv.reset() 
        mAPt.reset() 
        mAPi.reset() 
    with torch.no_grad():
        for batch, (img, (y1, y2, y3, y4)) in enumerate(dataloader):
            img, y1, y2, y3, y4 = img.cuda(), y1.cuda(), y2.cuda(), y3.cuda(), y4.cuda()            
            model.eval()  
            tool, verb, target, triplet = model(img)
            
            if not set_chlg_eval:
                cam_i, logit_i = tool
                cam_v, logit_v = verb
                cam_t, logit_t = target
                mAPi.update(y1.float().detach().cpu(), activation(logit_i).detach().cpu()) 
                mAPv.update(y2.float().detach().cpu(), activation(logit_v).detach().cpu())  
                mAPt.update(y3.float().detach().cpu(), activation(logit_t).detach().cpu())  
                
            mAP.update(y4.float().detach().cpu(), activation(triplet).detach().cpu()) 
            
    mAP.video_end() 
    
    if not set_chlg_eval:
        mAPv.video_end()
        mAPt.video_end()
        mAPi.video_end()

def weight_mgt(score, epoch):
    global benchmark
    if score > benchmark.item():
        torch.save(model.state_dict(), ckpt_path)
        benchmark = score
        print(f'   [GUARDADO EXITOSO] -> Nuevo récord en Época {epoch+1} | Archivo: {ckpt_path}', file=open(logfile, 'a+'))  
        return "increased"
    else:
        return "decreased"

#%% assign device and set debugger options
assign_gpu(gpu=gpu)
np.seterr(divide='ignore', invalid='ignore')
torch.autograd.set_detect_anomaly(False)

#%% data loading
dataset = dataloader.CholecT50( 
            dataset_dir=data_dir, 
            dataset_variant=dataset_variant,
            test_fold=kfold,
            augmentation_list=data_augmentations,
            )

train_dataset, val_dataset, test_dataset = dataset.build()
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_dataloader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)
 
test_dataloaders = []
for video_dataset in test_dataset:
    test_dataloader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    test_dataloaders.append(test_dataloader)
print("Dataset loaded ...")

#%% class weight balancing
class_weights = get_weight_balancing(case=dataset_variant)
if 'crossval' in dataset_variant:
    tool_weight   = class_weights[kfold]['tool']
    verb_weight   = class_weights[kfold]['verb']
    target_weight = class_weights[kfold]['target']
else:
    tool_weight   = class_weights['tool']
    verb_weight   = class_weights['verb']
    target_weight = class_weights['target']

tool_weight     = [0.93487068, 0.94234964, 0.93487068, 1.18448115, 1.02368339, 0.97974447]
verb_weight     = [0.60002400, 0.60002400, 0.60002400, 0.61682467, 0.67082683, 0.80163207, 0.70562823, 2.11208448, 2.69230769, 0.60062402]
target_weight   = [0.49752894, 0.52041527, 0.49752894, 0.51394739, 2.71899565, 1.75577963, 0.58509403, 1.25228034, 0.49752894, 2.42993134, 0.49802647, 0.87266576, 1.36074165, 0.50150917, 0.49802647]

#%% model
model = network.Rendezvous('resnet18', hr_output=hr_output, use_ln=use_ln).cuda()
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
benchmark   = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=False)
print("Model built ...")

#%% Loss
activation  = nn.Sigmoid()
loss_fn_i   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tool_weight).cuda())
loss_fn_v   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(verb_weight).cuda())
loss_fn_t   = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(target_weight).cuda())
loss_fn_ivt = nn.BCEWithLogitsLoss()

#%% evaluation metrics
mAP = ivtmetrics.Recognition(100)
mAP.reset_global()
if not set_chlg_eval:
    mAPi = ivtmetrics.Recognition(6)
    mAPv = ivtmetrics.Recognition(10)
    mAPt = ivtmetrics.Recognition(15)
    mAPi.reset_global()
    mAPv.reset_global()
    mAPt.reset_global()

#%% optimizer and lr scheduler
wp_lr           = [lr/power for lr in learning_rates]
module_i        = list(set(model.parameters()) - set(model.encoder.cagam.parameters()) - set(model.encoder.bottleneck.parameters()) - set(model.decoder.parameters()))
module_ivt      = list(set(model.encoder.bottleneck.parameters()).union(set(model.decoder.parameters())))
module_vt       = model.encoder.cagam.parameters()

optimizer_i     = torch.optim.SGD(module_i, lr=wp_lr[0], weight_decay=weight_decay)
scheduler_ia    = torch.optim.lr_scheduler.LinearLR(optimizer_i, start_factor=power, total_iters=warmups[0])
scheduler_ib    = torch.optim.lr_scheduler.ExponentialLR(optimizer_i, gamma=decay_rate)
scheduler_i     = torch.optim.lr_scheduler.SequentialLR(optimizer_i, schedulers=[scheduler_ia, scheduler_ib], milestones=[warmups[0]+1])

optimizer_vt    = torch.optim.SGD(module_vt, lr=wp_lr[1], weight_decay=weight_decay)
scheduler_vta   = torch.optim.lr_scheduler.LinearLR(optimizer_vt, start_factor=power, total_iters=warmups[1])
scheduler_vtb   = torch.optim.lr_scheduler.ExponentialLR(optimizer_vt, gamma=decay_rate)
scheduler_vt    = torch.optim.lr_scheduler.SequentialLR(optimizer_vt, schedulers=[scheduler_vta, scheduler_vtb], milestones=[warmups[1]+1])

optimizer_ivt   = torch.optim.SGD(module_ivt, lr=wp_lr[2], weight_decay=weight_decay)
scheduler_ivta  = torch.optim.lr_scheduler.LinearLR(optimizer_ivt, start_factor=power, total_iters=warmups[2])
scheduler_ivtb  = torch.optim.lr_scheduler.ExponentialLR(optimizer_ivt, gamma=decay_rate)
scheduler_ivt   = torch.optim.lr_scheduler.SequentialLR(optimizer_ivt, schedulers=[scheduler_ivta, scheduler_ivtb], milestones=[warmups[2]+1])

lr_schedulers   = [scheduler_i, scheduler_vt, scheduler_ivt]
optimizers      = [optimizer_i, optimizer_vt, optimizer_ivt]

#%% checkpoints/weights
if os.path.exists(ckpt_path):
    model.load_state_dict(torch.load(ckpt_path))
    resume_ckpt = ckpt_path
elif os.path.exists(pretrain_dir):
    pretrained_dict = torch.load(pretrain_dir)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model.state_dict().update(pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    resume_ckpt = pretrain_dir
print("Model's weight loaded ...")

#%% log config
header1 = "================================================================================================="
header2 = f"   INICIANDO SESIÓN DE ENTRENAMIENTO | Modelo: Rendezvous | Batch Físico: {batch_size} (Simulando 32)"
header3 = f"   Época de Arranque: {start_epoch} | Carga de Pesos: {resume_ckpt}"
header4 = "================================================================================================="
print(f"\n\n{header1}\n{header2}\n{header3}\n{header4}", file=open(logfile, 'a+'))

#%% run
if is_train:
    if start_epoch > 0:
        print(f"\n[INFO] Sincronizando calendario: Adelantando el reloj interno a la época {start_epoch}...", file=open(logfile, 'a+'))
        for _ in range(start_epoch):
            for sch in lr_schedulers:
                sch.step()

    for epoch in range(start_epoch, epochs):
        try:
            # LOG AMIGABLE - Inicio de época
            lrs_limpios = [f"{lr.get_last_lr()[0]:.5f}" for lr in lr_schedulers]
            print(f"\n-> ENTRENANDO ÉPOCA {epoch} | Tasa de Aprendizaje (LR): {lrs_limpios}", file=open(logfile, 'a+'))  
            
            train_loop(train_dataloader, model, activation, loss_fn_i, loss_fn_v, loss_fn_t, loss_fn_ivt, optimizers, lr_schedulers, epoch)

            # val
            if epoch % val_interval == 0:
                start = time.time()  
                mAP.reset_global()
                print(f"   Evaluando Examen Final de Época {epoch}...", file=open(logfile, 'a+'))
                test_loop(val_dataloader, model, activation, final_eval=False)
                
                # LOG AMIGABLE - Calificaciones Detalladas
                map_ivt = mAP.compute_video_AP('ivt', ignore_null=set_chlg_eval)['mAP']
                
                if not set_chlg_eval:
                    map_i = mAPi.compute_video_AP(ignore_null=False)['mAP']
                    map_v = mAPv.compute_video_AP(ignore_null=False)['mAP']
                    map_t = mAPt.compute_video_AP(ignore_null=False)['mAP']
                    print(f"   [CALIFICACIONES] -> Instr: {map_i:.4f} | Verbos: {map_v:.4f} | Órg: {map_t:.4f} | TRIPLETA (IVT): {map_ivt:.4f} | ETA: {(time.time() - start):.2f}s", file=open(logfile, 'a+'))
                else:
                    print(f"   [CALIFICACIONES] -> TRIPLETA (IVT): {map_ivt:.4f} | ETA: {(time.time() - start):.2f}s", file=open(logfile, 'a+'))
                
                behaviour = weight_mgt(map_ivt, epoch=epoch)
                
        except KeyboardInterrupt:
            print(f'\n>> [ALERTA] Entrenamiento pausado por el usuario en Época {epoch}. Archivos seguros.', file=open(logfile, 'a+'))   
            sys.exit(1)
    test_ckpt = ckpt_path