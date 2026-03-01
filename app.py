import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Auditoría CVC Pura: Motor de Detección Pro", layout="wide")

st.title("🔬 Módulo de Auditoría CVC Pura")
st.markdown("""
Esta herramienta ejecuta el **Motor de Detección Pura de Grado Pericial**. 
Su única función es escanear la hoja completa, ignorar la regla de los ejes y aislar todos los símbolos para auditoría visual.
- Cruz Azul = Centro Exacto.
- Bounding Box **Rojo** = Cuadrado (Fallado).
- Bounding Box **Verde** = Círculo (Visto).
""")

# ==========================================
# FUNCIONES DE VISIÓN (MOTOR PRO CORREGIDO)
# ==========================================

def find_and_clean_axes(thresh):
    """
    Encuentra los ejes, calcula el radio del campo y crea el borrador anti-ticks.
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
    
    # Calcular el radio exacto de los 60 grados midiendo la línea
    eje_derecho = lineas_h_puras[cy-5:cy+5, cx:]
    _, x_h = np.where(eje_derecho > 0)
    dist_60 = np.max(x_h) if len(x_h) > 0 else (ancho - cx)*0.75
    
    borrador_h_ticks = cv2.dilate(lineas_h_puras, np.ones((int(alto*0.015), 1), np.uint8))
    borrador_v_ticks = cv2.dilate(lineas_v_puras, np.ones((1, int(ancho*0.015)), np.uint8))
    borrador_anti_regla = cv2.add(borrador_h_ticks, borrador_v_ticks)
    
    return (cx, cy), borrador_anti_regla, dist_60


def classify_symbol(roi_bin):
    """
    Clasifica por erosión destructiva.
    """
    h, w = roi_bin.shape
    if cv2.countNonZero(roi_bin) < 5:
        return 'ignorar'
    
    k_size = max(2, int(min(w, h) * 0.40))
    kernel_erosion = np.ones((k_size, k_size), np.uint8)
    eroded_roi = cv2.erode(roi_bin, kernel_erosion, iterations=1)
    
    nucleo_area = cv2.countNonZero(eroded_roi)
    if nucleo_area / (float(
