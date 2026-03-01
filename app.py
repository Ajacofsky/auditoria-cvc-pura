import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="Auditoría CVC Pura: Motor de Detección Pro", layout="wide")

st.title("🔬 Módulo de Auditoría CVC Pura (Filtro 40°)")
st.markdown("""
Esta herramienta ejecuta el **Motor de Detección Pura**. 
Aísla la zona central y evalúa **ÚNICAMENTE los símbolos dentro de los 40 grados**.
- Cruz Azul = Centro Exacto.
- Anillo Naranja = Límite Legal de 40 Grados.
- Bounding Box **Rojo** = Cuadrado (Fallado).
- Bounding Box **Verde** = Círculo (Visto).
""")

# ==========================================
# FUNCIONES DE VISIÓN (MOTOR CON RADAR 40°)
# ==========================================

def find_and_clean_axes(thresh):
    """
    Encuentra los ejes, calcula la escala en píxeles y crea el borrador anti-ticks.
    """
    alto, ancho = thresh.shape
    
    zona_media_y = thresh[int(alto*0.25):int(alto*0.75), :]
    cy = np.argmax(np.sum(zona_media_y, axis=1)) + int(alto*0.25)
    
    zona_media_x = thresh[:, int(ancho*0.25):int(ancho*0.75)]
    cx = np.argmax(np.sum(zona_media_x, axis=0)) + int(ancho*0.25)
    
    k_len = max(20, int(ancho * 0.1))
    kernel_h_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    lineas_h_puras = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h_clean)
    
    k_len_v = max(20, int(alto * 0.1))
    kernel_v_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len_v))
    lineas_v_puras = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v_clean)
    
    # Calcular a cuántos píxeles están los 60 grados (final de la línea impresa)
    eje_derecho = lineas_h_puras[cy-5:cy+5, cx:]
    _, x_h = np.where(eje_derecho > 0)
    dist_60 = np.max(x_h) if len(x_h) > 0 else (ancho - cx)*0.75
    
    borrador_h_ticks = cv2.dilate(lineas_h_puras, np.ones((int(alto*0.015), 1), np.uint8))
    borrador_v_ticks = cv2.dilate(lineas_v_puras, np.ones((1, int(ancho*0.015)), np.uint8))
    borrador_anti_regla = cv2.add(borrador_h_ticks, borrador_v_ticks)
    
    return (cx, cy), borrador_anti_regla, dist_60


def classify_symbol(roi_bin):
    """
    Clasifica un símbolo mediante erosión destructiva.
    """
    h, w = roi_bin.shape
    if cv2.countNonZero(roi_bin) < 5:
        return 'ignorar'
    
    k_size = max(2, int(min(w, h) * 0.40))
    kernel_erosion = np.ones((k_size, k_size), np.uint8)
    eroded_roi = cv2.erode(roi_bin, kernel_erosion, iterations=1)
    
    nucleo_area = cv2.countNonZero(eroded_roi)
    if nucleo_area / (float(h*w)) > 0.05: 
        return 'fallado' 
    else:
        return 'visto'   


def detect_and_classify_symbols(img_bin, borrador_anti_regla, centro, pixels_por_10_grados):
    """
    Aísla y cuenta ÚNICAMENTE los símbolos dentro de los 40 grados.
    """
    alto, ancho = img_bin.shape
    img_auditoria = np.zeros((alto, ancho, 3), dtype=np.uint8) 
    img_auditoria[:,:] = [255, 255, 255] 
    
    campo_limpio = cv2.subtract(img_bin, borrador_anti_regla)
    simbolos_unidos = cv2.dilate(campo_limpio, np.ones((2,2), np.uint8))
    
    cuadrados_count = 0
    circulos_count = 0
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(simbolos_unidos, connectivity=8)
    
    area_min = (ancho * 0.002) ** 2
    area_max = (ancho * 0.02) ** 2
    cx, cy = centro
    
    for i in range(1, num_labels): 
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                     stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area_min < area < area_max and 0.4 < (w/float(h)) < 2.5:
            # Calcular el centro del símbolo
            px, py = x + w/2.0, y + h/2.0
            dx, dy = px - cx, py - cy
            
            # MAGIA GEOMÉTRICA: ¿A cuántos grados está este símbolo del centro?
            distancia_grados = (math.hypot(dx, dy) / pixels_por_10_grados) * 10.0
            
            # FILTRO: Solo pasamos a clasificar si está dentro de los 40 grados
            # (Le damos 1 grado de tolerancia por si el símbolo está mordiendo la línea)
            if distancia_grados <= 41.0:
                roi = campo_limpio[y:y+h, x:x+w]
                tipo = classify_symbol(roi)
                
                if tipo == 'fallado':
                    cuadrados_count += 1
                    cv2.rectangle(img_auditoria, (x, y), (x+w, y+h), (0, 0, 255), 2)
                elif tipo == 'visto':
                    circulos_count += 1
                    cv2.rectangle(img_auditoria, (x, y), (x+w, y+h), (0, 255, 0), 1)

    return img_auditoria, cuadrados_count, circulos_count

# ==========================================
# INTERFAZ WEB (`app.py`)
# ==========================================

archivo = st.file_uploader("Sube un estudio de CVC para testear el Área Central", type=["jpg", "jpeg", "png"])

if archivo is not None:
    with st.spinner("Mapeando el área central de 40 grados..."):
        
        nparr = np.frombuffer(archivo.getvalue(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        alto, ancho = thresh.shape

        # 1. Encontrar ejes y matemática de la escala
        centro, borrador_anti_regla, dist_60 = find_and_clean_axes(thresh)
        pixels_por_10_grados = float(dist_60 / 6.0)
        
        # 2. Detección Exclusiva 40°
        img_auditoria_bin, t_cuad, t_circ = detect_and_classify_symbols(thresh, borrador_anti_regla, centro, pixels_por_10_grados)
        
        # 3. Fusión visual (Pintar sobre la imagen original)
        img_final = img.copy()
        for i in range(3):
            mask = img_auditoria_bin[:,:,i] != 255
            img_final[mask, i] = img_auditoria_bin[mask, i]
            
        # Dibujar Centro (Azul)
        cv2.line(img_final, (0, centro[1]), (ancho, centro[1]), (255, 0, 0), 1)
        cv2.line(img_final, (centro[0], 0), (centro[0], alto), (255, 0, 0), 1)
        
        # Dibujar Frontera Legal de 40° (Naranja grueso)
        radio_40_px = int(4.0 * pixels_por_10_grados)
        cv2.circle(img_final, centro, radio_40_px, (0, 165, 255), 3)

        # MOSTRAR RESULTADOS
        col1, col2 = st.columns([3, 1])
        with col1:
            img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(img_rgb), caption="Auditoría Visual (Anillo Naranja = 40°)", use_container_width=True)
        with col2:
            st.markdown("---")
            st.markdown("### 🔬 Conteo Área 40°")
            
            data = {"Cuadrados": [t_cuad], "Círculos": [t_circ]}
            st.bar_chart(data)
            
            st.metric("Cuadrados (Rojos)", t_cuad)
            st.metric("Círculos (Verdes)", t_circ)
            st.write("---")
            st.markdown("""
            **Verificación Pericial:**
            La computadora **ignora** cualquier símbolo u objeto (texto) que se encuentre por fuera del anillo naranja de los 40 grados, garantizando la zona legal del cálculo.
            """)
