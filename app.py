import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image

st.set_page_config(page_title="Detector de CVC PRO: Motor Oro v2", layout="wide")

st.title("🔬 Módulo de Prueba: Detección Pura (Anti-ticks PRO)")
st.markdown("""
**Programa Activo:** Motor de Detección Núcleo-Físico con **Borrador Asimétrico Mágico de Ejes**.
Su única función es escanear la hoja completa, ignorar la regla de los ejes y aislar todos los símbolos para auditoría visual.
- Cruz Azul = Centro Exacto.
- Bounding Box **Rojo** = Cuadrado (Fallado).
- Bounding Box **Verde** = Círculo (Visto).
- Las marcas de la regla impresa (ticks) están barridas asimétricamente.
""")

def detectar_simbolos(image_bytes):
    # 1. Cargar imagen y convertir a escala de grises
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_debug = img.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Binarización (Tinta negra = Blanco 255, Papel = Negro 0)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    alto, ancho = thresh.shape
    
    # --- A. ENCONTRAR EL CENTRO EXACTO (MÉTODO DE PROYECCIÓN) ---
    # Ignoramos el 25% superior e inferior para que el texto/leyenda no interfiera
    zona_media_y = thresh[int(alto*0.25):int(alto*0.75), :]
    cy = np.argmax(np.sum(zona_media_y, axis=1)) + int(alto*0.25) # Fila con más tinta (Eje X)
    
    zona_media_x = thresh[:, int(ancho*0.25):int(ancho*0.75)]
    cx = np.argmax(np.sum(zona_media_x, axis=0)) + int(ancho*0.25) # Columna con más tinta (Eje Y)
    
    # Dibujar cruz AZUL para demostrar que encontramos el centro real
    cv2.line(img_debug, (0, cy), (ancho, cy), (255, 0, 0), 1)
    cv2.line(img_debug, (cx, 0), (cx, alto), (255, 0, 0), 1)

    # --- B. DETECCIÓN DE SÍMBOLOS CON BORRADOR MÁGICO ASIMÉTRICO DE EJE ---
    
    # Encontramos las líneas rectas puras de la grilla
    k_len = max(20, int(ancho * 0.1)) # Kernel largo para líneas
    kernel_h_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (k_len, 1))
    kernel_v_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_len))
    lineas_h = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_h_clean)
    lineas_v = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_v_clean)
    
    # MEJORA: Engrosamiento asimétrico dirigido para "barrer" los ticks de la regla
    # Usamos un "borrador" vertical para la línea horizontal
    grilla_h_ancha = cv2.dilate(lineas_h, np.ones((int(alto*0.015), 1), np.uint8)) 
    # Usamos un "borrador" horizontal para la línea vertical
    grilla_v_ancha = cv2.dilate(lineas_v, np.ones((1, int(ancho*0.015)), np.uint8)) 
    
    # Combinamos para obtener la grilla engrosada (Borrador completo anti-regla)
    grilla_engrosada = cv2.add(grilla_h_ancha, grilla_v_ancha)
    
    # Restamos físicamente las líneas y los ticks de la imagen binarizada
    campo_limpio = cv2.subtract(thresh, grilla_engrosada)
    
    # Pequeña dilatación para unir símbolos que hayan sido cortados por la resta
    simbolos_unidos = cv2.dilate(campo_limpio, np.ones((2,2), np.uint8))
    
    # Encontrar contornos
    contornos, _ = cv2.findContours(simbolos_unidos, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtros de tamaño dinámicos basados en la resolución de la imagen
    area_min = (ancho * 0.002) ** 2
    area_max = (ancho * 0.02) ** 2
    
    cuadrados_encontrados = 0
    circulos_encontrados = 0
    
    for cnt in contornos:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect_ratio = w / float(h)
        
        # Filtro de forma: si es demasiado alargado, no es un símbolo
        if area_min < area < area_max and 0.4 < aspect_ratio < 2.5:
            # ROI en la imagen LIMPIA binarizada (sin líneas ni ticks)
            roi = campo_limpio[y:y+h, x:x+w]
            
            # Clasificación por Erosión Destructiva
            # Creamos una "lija" cuyo tamaño depende de la caja misma (40% de su tamaño)
            k_size = max(2, int(min(w, h) * 0.40))
            kernel_erosion = np.ones((k_size, k_size), np.uint8)
            
            # Aplicamos la lija: Si es línea fina desaparece. Si es bloque sobrevive.
            eroded_roi = cv2.erode(roi, kernel_erosion, iterations=1)
            
            if cv2.countNonZero(eroded_roi) > 0:
                cuadrados_encontrados += 1
                cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 0, 255), 2) # Caja Roja
            else:
                circulos_encontrados += 1
                cv2.rectangle(img_debug, (x, y), (x+w, y+h), (0, 255, 0), 1) # Caja Verde

    return img_debug, cuadrados_encontrados, circulos_encontrados

# --- INTERFAZ ---
archivo = st.file_uploader("Sube el CVC para probar el Borrador Asimétrico Mágico", type=["jpg", "jpeg", "png"])

if archivo is not None:
    with st.spinner("Escaneando con el nuevo motor PRO anti-ticks..."):
        img_res, total_cuadrados, total_circulos = detectar_simbolos(archivo.getvalue())
        
        col1, col2 = st.columns([3, 1])
        with col1:
            img_rgb = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
            st.image(Image.fromarray(img_rgb), caption="Auditoría Visual (Ejes Ignorados)", use_container_width=True)
        with col2:
            st.metric("Cuadrados (Rojos)", total_cuadrados)
            st.metric("Círculos (Verdes)", total_circulos)
            st.write("---")
            st.write("**Modo de Auditoría:**")
            st.write("1. Cruz azul en el centro exacto.")
            st.write("2. Cuadrados encerrados en ROJO.")
            st.write("3. Círculos encerrados en VERDE.")
            st.write("4. **Las marcas de la regla impresa (ticks) deberían estar completamente ignoradas.**")
