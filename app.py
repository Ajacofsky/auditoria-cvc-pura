import streamlit as st
import cv2
import numpy as np
import math
import re
import base64
import os
import tempfile
from datetime import datetime
from PIL import Image
from fpdf import FPDF
import pytesseract

st.set_page_config(page_title="Calculadora Pericial CVC", layout="wide")

st.title("⚖️ Calculadora Pericial CVC (Área Central 40°)")
st.markdown("""
**Suite de Dictamen Médico-Legal**
- Motor de detección blindado (Universo de 104 puntos).
- Lector óptico de datos y Reporte fotográfico de 1 página.
""")

# ==========================================
# 🔒 MOTOR DE LECTURA (OCR MEJORADO)
# ==========================================
def extraer_nombre(img_gray):
    """Escanea el encabezado intentando salvar el formato de matriz de puntos."""
    try:
        alto, ancho = img_gray.shape
        header_img = img_gray[0:int(alto*0.20), :]
        
        # Binarización Otsu para mejorar contraste de letras borrosas
        _, thresh_ocr = cv2.threshold(header_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        texto = pytesseract.image_to_string(thresh_ocr, config='--psm 6')
        
        # Buscar NOMBRE y agarrar letras/espacios hasta toparse con doble espacio o palabras clave
        match = re.search(r'NOMBRE[\s:.]+([A-Z\s]+?)(?:\s{2,}|\n|DERECHO|IZQUIERDO|ID)', texto, re.IGNORECASE)
        if match:
            nombre_limpio = match.group(1).strip()
            # Eliminar basura si leyó mal
            nombre_limpio = re.sub(r'[^A-Z\s]', '', nombre_limpio) 
            return nombre_limpio
        return ""
    except Exception as e:
        return ""

# ==========================================
# 🔒 MOTOR DE VISIÓN BLINDADO 
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
    if cv2.countNonZero(roi_bin) < 5: return 'ignorar'
    k_size = max(2, int(min(w, h) * 0.40))
    eroded_roi = cv2.erode(roi_bin, np.ones((k_size, k_size), np.uint8), iterations=1)
    if cv2.countNonZero(eroded_roi) / (float(h*w)) > 0.05: return 'fallado' 
    return 'visto'   

def detect_and_classify_symbols(img_bin, borrador_anti_regla, centro, pixels_por_10_grados):
    alto, ancho = img_bin.shape
    img_auditoria = np.zeros((alto, ancho, 3), dtype=np.uint8) 
    img_auditoria[:,:] =
