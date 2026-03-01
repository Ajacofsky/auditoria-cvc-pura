import streamlit as st
import cv2
import numpy as np
import math
import base64
import os
import tempfile
from PIL import Image
from fpdf import FPDF

st.set_page_config(page_title="Calculadora Pericial CVC", layout="wide")

st.title("⚖️ Calculadora Pericial CVC (Área Central 40°)")
st.markdown("""
**Suite de Dictamen Médico-Legal**
- Motor de detección blindado (Universo de 104 puntos).
- Reporte Fotográfico 100% fiel al documento original.
- Exportación a PDF en una sola página, limpio y ordenado.
""")

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
    img_auditoria[:,:] = [255, 255, 255] 
    
    campo_limpio = cv2.subtract(img_bin, borrador_anti_regla)
    grosor_pegamento = max(3, int(alto*0.004)) + 2
    simbolos_unidos = cv2.dilate(campo_limpio, np.ones((grosor_pegamento, grosor_pegamento), np.uint8))
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(simbolos_unidos, connectivity=8)
    area_min, area_max = (ancho * 0.002) ** 2, (ancho * 0.02) ** 2
    cx, cy = centro
    cuadrados_count, circulos_count = 0, 0
    
    for i in range(1, num_labels): 
        x, y, w, h, area = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if area_min < area < area_max and 0.4 < (w/float(h)) < 2.5:
            px, py = x + w/2.0, y + h/2.0
            if (math.hypot(px - cx, py - cy) / pixels_por_10_grados) * 10.0 <= 41.0:
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
# GENERADOR DE PDF (MÁS PEQUEÑAS Y SIN SOLAPAMIENTO)
# ==========================================
def generar_pdf_moderno(incap_od, grados_od, img_od_orig, incap_oi, grados_oi, img_oi_orig, incap_total, modo):
    pdf = FPDF()
    pdf.add_page()
    
    # 1. ENCABEZADO MINIMALISTA
    pdf.set_fill_color(41, 64, 115) 
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", 'B', 15)
    pdf.cell(0, 16, "  DICTAMEN PERICIAL - CAMPO VISUAL COMPUTARIZADO", 0, 1, 'C', fill=True)
    pdf.ln(5) # Poco espacio arriba

    # 2. IMÁGENES ORIGINALES (Mucho más chicas)
    y_images = pdf.get_y()
    
    if modo == "Bilateral (OD y OI)":
        if img_od_orig is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_od:
                cv2.imwrite(tmp_od.name, img_od_orig)
                # Reducido a 75mm de ancho para que la altura no invada abajo
                pdf.image(tmp_od.name, x=25, y=y_images, w=75) 
            os.remove(
