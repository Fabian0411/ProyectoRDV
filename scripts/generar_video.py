import os
import cv2
import torch
import glob
import numpy as np
import network
from torchvision import transforms

print("Iniciando el Quirófano Virtual (Demo de Validación Automática)...")

# ==========================================
# 1. CONFIGURACIÓN DE RUTAS
# ==========================================
carpeta_entrada = r"c:/Users/Nan/OneDrive/Documentos/Servicio Social/LSTM/CholecT50/videos/VID80"  # <--- CAMBIA ESTO
video_salida    = r"./demo_validacion_ia.mp4"
ckpt_path       = r"./__checkpoint__/run_0/rendezvous_l2_cholectcholect45-crossval_k1_batchnorm_lowres.pth"

# RUTAS DE TUS HOJAS DE RESPUESTAS (GROUND TRUTH)
ruta_txt_tools   = r"C:/Users/Nan/OneDrive/Documentos/Servicio Social/LSTM/CholecT50/instrument/VID80.txt"       # <--- CAMBIA ESTO
ruta_txt_verbs   = r"C:/Users/Nan/OneDrive/Documentos/Servicio Social/LSTM/CholecT50/verb/VID80.txt"             # <--- CAMBIA ESTO
ruta_txt_targets = r"C:/Users/Nan/OneDrive/Documentos/Servicio Social/LSTM/CholecT50/target/VID80.txt"           # <--- CAMBIA ESTO

tool_names = ['Grasper', 'Bipolar', 'Hook', 'Scissors', 'Clipper', 'Irrigator']
verb_names = ['Grasp', 'Retract', 'Dissect', 'Coagulate', 'Clip', 'Cut', 'Aspirate', 'Irrigate', 'Pack', 'Null']
target_names = ['Gallbladder', 'Cystic Plate', 'Cystic Duct', 'Cystic Artery', 'Cystic Pedicle', 'Blood Vessel', 'Fluid', 'Adhesion', 'Omentum', 'Liver', 'Gut', 'Specimen Bag', 'Abdominal Wall', 'Gauze', 'Null']

# ==========================================
# 2. FUNCIÓN PARA LEER LAS HOJAS DE RESPUESTAS
# ==========================================
def cargar_ground_truth(ruta_archivo, num_clases):
    gt_dict = {}
    if not os.path.exists(ruta_archivo):
        print(f"ADVERTENCIA: No se encontró el archivo {ruta_archivo}")
        return gt_dict
    
    with open(ruta_archivo, 'r') as f:
        for linea in f:
            datos = linea.strip().split(',')
            if len(datos) > num_clases:
                frame_id = int(datos[0])
                # Convertimos los '0' y '1' a enteros (saltándonos el frame_id)
                etiquetas = [int(x) for x in datos[1:]]
                gt_dict[frame_id] = etiquetas
    return gt_dict

print("Cargando hojas de respuestas del cirujano...")
gt_tools = cargar_ground_truth(ruta_txt_tools, len(tool_names))
gt_verbs = cargar_ground_truth(ruta_txt_verbs, len(verb_names))
gt_targets = cargar_ground_truth(ruta_txt_targets, len(target_names))

# ==========================================
# 3. PREPARAR EL CEREBRO DE LA IA
# ==========================================
print("Cargando el cerebro de la Época 44...")
model = network.Rendezvous('resnet18', hr_output=False, use_ln=False).cuda()
model.load_state_dict(torch.load(ckpt_path))
model.eval()
activation = torch.nn.Sigmoid()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 224)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 4. PREPARAR EL CREADOR DE VIDEO
# ==========================================
frames = glob.glob(os.path.join(carpeta_entrada, "*.*"))
frames = [f for f in frames if f.endswith(('.png', '.jpg', '.jpeg'))]
frames.sort()

if not frames:
    print("¡Error! No encontré ninguna imagen en la carpeta de entrada.")
    exit()

frame_prueba = cv2.imread(frames[0])
alto_original, ancho_original = frame_prueba.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# A 15 FPS para que tu profesor pueda leer los textos con calma
grabador = cv2.VideoWriter(video_salida, fourcc, 15.0, (ancho_original, alto_original))

# ==========================================
# 5. BUCLE PRINCIPAL (ANÁLISIS FRAME POR FRAME)
# ==========================================
with torch.no_grad():
    for idx, frame_path in enumerate(frames):
        img_bgr = cv2.imread(frame_path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        img_tensor = transform(img_rgb).unsqueeze(0).cuda()
        tool, verb, target, triplet = model(img_tensor)
        
        cam_i, logit_i = tool
        _, logit_v = verb
        _, logit_t = target
        
        prob_i = activation(logit_i[0]).cpu().numpy()
        prob_v = activation(logit_v[0]).cpu().numpy()
        prob_t = activation(logit_t[0]).cpu().numpy()
        
        imagen_final = img_bgr.copy()
        
        # --- DIBUJAR CUADRADO DE ENFOQUE (Solo para herramientas) ---
        herramienta_top_idx = np.argmax(prob_i)
        if prob_i[herramienta_top_idx] > 0.5:
            mapa_atencion = cam_i[0, herramienta_top_idx].cpu().numpy()
            max_y_small, max_x_small = np.unravel_index(np.argmax(mapa_atencion), mapa_atencion.shape)
            
            factor_escala_w = ancho_original / mapa_atencion.shape[1]
            factor_escala_h = alto_original / mapa_atencion.shape[0]
            max_x_original = int(max_x_small * factor_escala_w)
            max_y_original = int(max_y_small * factor_escala_h)
            
            tamanio_caja = 100
            x_tl = max(0, max_x_original - tamanio_caja // 2)
            y_tl = max(0, max_y_original - tamanio_caja // 2)
            x_br = min(ancho_original - 1, max_x_original + tamanio_caja // 2)
            y_br = min(alto_original - 1, max_y_original + tamanio_caja // 2)
            
            cv2.rectangle(imagen_final, (x_tl, y_tl), (x_br, y_br), (0, 0, 255), thickness=4)

        # ==========================================
        # 6. LÓGICA DE VALIDACIÓN (IA vs MÉDICO REAL)
        # ==========================================
        # Obtener las respuestas reales para ESTE frame (asumiendo que idx = frame_id)
        # Si tu carpeta no empieza desde el frame 0, tendríamos que extraer el número del nombre del archivo.
        # Por ahora asumimos que el índice de la foto (0, 1, 2) coincide con el TXT.
        real_tool_bin = gt_tools.get(idx, [0]*len(tool_names))
        real_verb_bin = gt_verbs.get(idx, [0]*len(verb_names))
        real_target_bin = gt_targets.get(idx, [0]*len(target_names))

        # Determinar nombres reales
        real_tools_names = [tool_names[i] for i, val in enumerate(real_tool_bin) if val == 1]
        real_verbs_names = [verb_names[i] for i, val in enumerate(real_verb_bin) if val == 1]
        real_targets_names = [target_names[i] for i, val in enumerate(real_target_bin) if val == 1]

        str_real_tool = "DR: " + (", ".join(real_tools_names) if real_tools_names else "Ninguna")
        str_real_verb = "DR: " + (", ".join(real_verbs_names) if real_verbs_names else "Ninguna")
        str_real_target = "DR: " + (", ".join(real_targets_names) if real_targets_names else "Ninguno")

        # Determinar predicción TOP de la IA
        pred_tool_name = f"IA: {tool_names[herramienta_top_idx]} ({prob_i[herramienta_top_idx]*100:.1f}%)" if prob_i[herramienta_top_idx] > 0.5 else "IA: Ninguna"
        
        verb_top_idx = np.argmax(prob_v)
        pred_verb_name = f"IA: {verb_names[verb_top_idx]} ({prob_v[verb_top_idx]*100:.1f}%)" if prob_v[verb_top_idx] > 0.5 else "IA: Ninguna"
        
        target_top_idx = np.argmax(prob_t)
        pred_target_name = f"IA: {target_names[target_top_idx]} ({prob_t[target_top_idx]*100:.1f}%)" if prob_t[target_top_idx] > 0.5 else "IA: Ninguno"

        # EVALUACIÓN (¿Acertó la IA?)
        # Es correcto si predijo algo y ese algo está en la lista real, o si no predijo nada y la lista real está vacía.
        correcto_tool = (real_tool_bin[herramienta_top_idx] == 1) if prob_i[herramienta_top_idx] > 0.5 else (sum(real_tool_bin) == 0)
        correcto_verb = (real_verb_bin[verb_top_idx] == 1) if prob_v[verb_top_idx] > 0.5 else (sum(real_verb_bin) == 0)
        correcto_target = (real_target_bin[target_top_idx] == 1) if prob_t[target_top_idx] > 0.5 else (sum(real_target_bin) == 0)

        # ==========================================
        # 7. DIBUJAR EN PANTALLA
        # ==========================================
        fuente = cv2.FONT_HERSHEY_SIMPLEX
        escala_dr = 0.6 # Texto del doctor más pequeño
        escala_ia = 0.8 # Texto de IA más grande
        
        color_ok = (0, 255, 0)   # Verde
        color_err = (0, 0, 255)  # Rojo
        color_dr = (255, 255, 255) # Blanco

        # Dibujar Fondo negro para los textos (mejor legibilidad)
        cv2.rectangle(imagen_final, (10, 10), (550, 230), (0,0,0), -1)

        # Textos del Médico (Blanco)
        cv2.putText(imagen_final, str_real_tool, (20, 40), fuente, escala_dr, color_dr, 1)
        cv2.putText(imagen_final, str_real_verb, (20, 110), fuente, escala_dr, color_dr, 1)
        cv2.putText(imagen_final, str_real_target, (20, 180), fuente, escala_dr, color_dr, 1)

        # Textos de la IA (Verde o Rojo dependiendo de si acertó)
        color_t = color_ok if correcto_tool else color_err
        color_v = color_ok if correcto_verb else color_err
        color_tar = color_ok if correcto_target else color_err

        texto_t_final = pred_tool_name + (" [OK]" if correcto_tool else " [ERROR]")
        texto_v_final = pred_verb_name + (" [OK]" if correcto_verb else " [ERROR]")
        texto_tar_final = pred_target_name + (" [OK]" if correcto_target else " [ERROR]")

        cv2.putText(imagen_final, texto_t_final, (20, 70), fuente, escala_ia, color_t, 2)
        cv2.putText(imagen_final, texto_v_final, (20, 140), fuente, escala_ia, color_v, 2)
        cv2.putText(imagen_final, texto_tar_final, (20, 210), fuente, escala_ia, color_tar, 2)

        grabador.write(imagen_final)
        
        if idx % 50 == 0:
            print(f"Renderizando: {idx}/{len(frames)} frames...")

grabador.release()
print(f"\n¡Demo de Validación Completada! Tu video se guardó en: {video_salida}")