¡Esa es la mentalidad exacta de un auditor de grado pericial! Regla número uno: el motor de visión que ya funciona se "blinda" y no se toca.

Para cumplir tu condición al pie de la letra, he dejado la matemática de detección, el dibujo de los 40 grados y el bisturí de los ejes absolutamente intactos. Ni una sola coma ha cambiado en la forma en que la computadora lee la imagen.

Lo que hice fue agregar una "capa humana" al final del proceso en la interfaz web.

La Mejora Segura: "El Panel de Corrección Pericial"
A partir de ahora, la aplicación funcionará en dos pasos:

La Propuesta de la Máquina: El programa escaneará la imagen y te dirá, por ejemplo: "Encontré 97 cuadrados y 19 círculos".

Tu Veredicto (La novedad): Justo debajo, aparecerá un nuevo panel con contadores manuales. Estos contadores cargarán los números de la máquina por defecto, pero tú podrás sumar o restar puntos haciendo clic en los botones de "+" o "-".

Si ves que la máquina omitió 4 círculos en el eje, simplemente vas al contador de círculos, le sumas 4, y el cálculo final de incapacidad se actualizará instantáneamente en tiempo real basándose en TUS números, no en los de la máquina.

El Código Seguro (Motor Intacto + Panel Pericial)
Copia este código, guárdalo en tu app.py y súbelo a GitHub. No hay riesgo de romper la detección porque esa parte del código está idéntica a la versión que aprobaste.

Python
import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="Calculadora Pericial CVC", layout="wide")

st.title("⚖️ Calculadora Pericial CVC (Área Central 40°)")
st.markdown("""
**Motor de Detección con Auditoría Humana**
La máquina aísla y cuenta los símbolos dentro de los 40 grados. **El perito tiene la última palabra para ajustar el conteo final.**
- **Cruz Azul:** Centro Exacto de Fijación.
- **Anillo Naranja:** Límite Pericial de 40 Grados.
- **Cajas:** Rojo (Fallado) / Verde (Visto).
""")

# ==========================================
# 🔒 MOTOR DE VISIÓN BLINDADO (NO SE TOCÓ)
# ==========================================

def find_and_clean_axes(thresh):
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
    
    eje_derecho = lineas_h_puras[cy-5:cy+5, cx:]
    _, x_h = np.where(eje_derecho > 0)
    dist_60 = np.max(x_h) if len(x_h) > 0 else (ancho - cx)*0.75
    
    grosor_fino_h = max(3, int(alto*0.004))
    grosor_fino_v = max(3, int(ancho*0.004))
    borrador_h_ticks = cv2.dilate(lineas_h_puras, np.ones((grosor_fino_h, 1), np.uint8))
    borrador_v_ticks = cv2.dilate(lineas_v_puras, np.ones((1, grosor_fino_v), np.uint8))
    borrador_anti_regla = cv2.add(borrador_h_ticks, borrador_v_ticks)
    
    return (cx, cy), borrador_anti_regla, dist_60

def classify_symbol(roi_bin):
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
    alto, ancho = img_bin.shape
    img_auditoria = np.zeros((alto, ancho, 3), dtype=np.uint8) 
    img_auditoria[:,:] = [255, 255, 255] 
    
    campo_limpio = cv2.subtract(img_bin, borrador_anti_regla)
    
    grosor_pegamento = max(3, int(alto*0.004)) + 2
    simbolos_unidos = cv2.dilate(campo_limpio, np.ones((grosor_pegamento, grosor_pegamento), np.uint8))
    
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
            px, py = x + w/2.0, y + h/2.0
            dx, dy = px - cx, py - cy
            
            distancia_grados = (math.hypot(dx, dy) / pixels_por_10_grados) * 10.0
            
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

archivo = st.file_uploader("Sube un estudio de CVC", type=["jpg", "jpeg", "png"])

if archivo is not None:
    with st.spinner("Procesando área legal de 40 grados..."):
        
        nparr = np.frombuffer(archivo.getvalue(), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
        alto, ancho = thresh.shape

        centro, borrador_anti_regla, dist_60 = find_and_clean_axes(thresh)
        pixels_por_10_grados = float(dist_60 / 6.0)
        
        img_auditoria_bin, t_cuad, t_circ = detect_and_classify_symbols(thresh, borrador_anti_regla, centro, pixels_por_10_grados)
        
        img_final = img.copy()
        for i in range(3):
            mask = img_auditoria_bin[:,:,i] != 255
            img_final[mask, i] = img_auditoria_bin[mask, i]
            
        cv2.line(img_final, (0, centro[1]), (ancho, centro[1]), (255, 0, 0), 1)
        cv2.line(img_final, (centro[0], 0), (centro[0], alto), (255, 0, 0), 1)
        
        radio_40_px = int(4.0 * pixels_por_10_grados)
        cv2.circle(img_final, centro, radio_40_px, (0, 165, 255), 3)

        # ------------------------------------------
        # MOSTRAR RESULTADOS
        # ------------------------------------------
        col1, col2 = st.columns([3, 2])
        
        with col1:
            img_rgb = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(img_rgb), caption="Mapa Visual (Anillo Naranja = 40°)", use_container_width=True)
            
        with col2:
            st.markdown("### 1️⃣ Detección de la Máquina")
            st.info("La computadora propone el siguiente conteo inicial:")
            c1, c2 = st.columns(2)
            c1.metric("Cuadrados Detectados", t_cuad)
            c2.metric("Círculos Detectados", t_circ)
            
            st.markdown("---")
            
            # NUEVO: PANEL DE CORRECCIÓN MANUAL
            st.markdown("### 2️⃣ Panel de Corrección Pericial")
            st.write("Ajuste los valores si la máquina omitió símbolos por ruido de impresión.")
            
            adj1, adj2 = st.columns(2)
            with adj1:
                cuadrados_final = st.number_input("Cuadrados (Fallados) Reales:", min_value=0, max_value=104, value=t_cuad, step=1)
            with adj2:
                circulos_final = st.number_input("Círculos (Vistos) Reales:", min_value=0, max_value=104, value=t_circ, step=1)
            
            st.markdown("---")
            
            # CÁLCULO LEGAL SOBRE LOS NÚMEROS VALIDADOS
            base_calculo = 104.0 
            grados_no_vistos = (cuadrados_final / base_calculo) * 320.0
            incapacidad_porcentaje = (grados_no_vistos / 320.0) * 100 * 0.25
            
            st.markdown("### 3️⃣ Informe Matemático Definitivo")
            st.metric("Grados No Vistos (Validado)", f"{grados_no_vistos:.1f}°", f"Base: {cuadrados_final} cuadros de 104")
            st.metric("Incapacidad Unilateral", f"{incapacidad_porcentaje:.2f}%", "Basado en
