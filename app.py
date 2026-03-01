import streamlit as st
import cv2
import numpy as np
import math
from PIL import Image
from fpdf import FPDF
import base64

st.set_page_config(page_title="Calculadora Pericial CVC", layout="wide")

st.title("⚖️ Calculadora Pericial CVC (Área Central 40°)")
st.markdown("""
**Motor de Detección con Auditoría Humana y Reporte PDF**
- Aísla la zona central (Universo de 104 puntos).
- El perito tiene la última palabra sobre el conteo final.
- Cálculo automático de bilateralidad y exportación legal.
""")

# ==========================================
# 🔒 MOTOR DE VISIÓN BLINDADO 104
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
    
    if cv2.countNonZero(eroded_roi) / (float(h*w)) > 0.05: 
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
# GENERADOR DE PDF 
# ==========================================

def generar_pdf_base64(incap_od, incap_oi, incap_total, modo):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "DICTAMEN PERICIAL - CAMPO VISUAL COMPUTARIZADO", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "Metodologia: Analisis del area central (40 grados), base 104 estimulos.", ln=True)
    pdf.ln(5)
    
    if modo == "Unilateral (1 Ojo)":
        ojo_str = "Derecho (OD)" if incap_od > 0 else "Izquierdo (OI)"
        incap_val = incap_od if incap_od > 0 else incap_oi
        pdf.cell(0, 10, f"Ojo Evaluado: {ojo_str}", ln=True)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"Incapacidad Unilateral Validada: {incap_val:.2f}%", ln=True)
    else:
        pdf.cell(0, 10, f"Incapacidad Unilateral Ojo Derecho (OD): {incap_od:.2f}%", ln=True)
        pdf.cell(0, 10, f"Incapacidad Unilateral Ojo Izquierdo (OI): {incap_oi:.2f}%", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, f"Suma Aritmetica (OD + OI): {(incap_od + incap_oi):.2f}%", ln=True)
        pdf.cell(0, 10, f"Factor de Bilateralidad Aplicado: x 1.5", ln=True)
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, f"INCAPACIDAD TOTAL BILATERAL: {incap_total:.2f}%", ln=True)
        
    pdf.ln(30)
    pdf.set_font("Arial", '', 12)
    pdf.cell(0, 10, "_____________________________________________________", ln=True, align='C')
    pdf.cell(0, 10, "Firma y Sello del Perito Medico", ln=True, align='C')
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    b64 = base64.b64encode(pdf_bytes).decode()
    return b64

# ==========================================
# INTERFAZ WEB (`app.py`)
# ==========================================

modo_evaluacion = st.radio("Seleccione el Tipo de Evaluación:", ["Unilateral (1 Ojo)", "Bilateral (OD y OI)"], horizontal=True)
st.divider()

def procesar_panel_ojo(titulo_ojo, key_suffix):
    archivo = st.file_uploader(f"Subir estudio - {titulo_ojo}", type=["jpg", "jpeg", "png"], key=f"file_{key_suffix}")
    incapacidad_final = 0.0
    
    if archivo is not None:
        with st.spinner(f"Escaneando {titulo_ojo}..."):
            nparr = np.frombuffer(archivo.getvalue(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            
            centro, borrador_anti_regla, dist_60 = find_and_clean_axes(thresh)
            pixels_por_10_grados = float(dist_60 / 6.0)
            
            img_auditoria_bin, t_cuad, t_circ = detect_and_classify_symbols(thresh, borrador_anti_regla, centro, pixels_por_10_grados)
            
            img_final = img.copy()
            for i in range(3):
                mask = img_auditoria_bin[:,:,i] != 255
                img_final[mask, i] = img_auditoria_bin[mask, i]
                
            cv2.circle(img_final, centro, int(4.0 * pixels_por_10_grados), (0, 165, 255), 3)

            # INTERFAZ DEL PANEL
            st.image(Image.fromarray(cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)), caption=f"Auditoría {titulo_ojo}", use_container_width=True)
            
            st.markdown(f"**Corrección Pericial - {titulo_ojo}**")
            col_a, col_b = st.columns(2)
            with col_a:
                cuadrados_final = st.number_input("Cuadrados Reales:", min_value=0, max_value=104, value=t_cuad, step=1, key=f"cuad_{key_suffix}")
            with col_b:
                circulos_final = st.number_input("Círculos Reales:", min_value=0, max_value=104, value=t_circ, step=1, key=f"circ_{key_suffix}")
                
            grados_no_vistos = (cuadrados_final / 104.0) * 320.0
            incapacidad_final = (grados_no_vistos / 320.0) * 100 * 0.25
            
            st.metric(f"Incapacidad {titulo_ojo}", f"{incapacidad_final:.2f}%")
            
    return incapacidad_final

# Layout de columnas
if modo_evaluacion == "Unilateral (1 Ojo)":
    incap_od = procesar_panel_ojo("Ojo Evaluado", "unico")
    incap_oi = 0.0
else:
    col_izq, col_der = st.columns(2)
    with col_izq:
        incap_od = procesar_panel_ojo("Ojo Derecho (OD)", "od")
    with col_der:
        incap_oi = procesar_panel_ojo("Ojo Izquierdo (OI)", "oi")

st.divider()

# ==========================================
# DICTAMEN FINAL Y PDF
# ==========================================
st.header("📋 Dictamen Legal Final")

incap_total_bilateral = 0.0

if modo_evaluacion == "Bilateral (OD y OI)":
    if incap_od > 0 or incap_oi > 0:
        suma_aritmetica = incap_od + incap_oi
        incap_total_bilateral = suma_aritmetica * 1.5
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Suma Aritmética", f"{suma_aritmetica:.2f}%")
        c2.metric("Factor Bilateralidad", "x 1.5")
        c3.metric("INCAPACIDAD TOTAL", f"{incap_total_bilateral:.2f}%")
    else:
        st.info("Suba al menos una imagen para ver el cálculo final.")
else:
    if incap_od > 0:
        st.metric("INCAPACIDAD UNILATERAL", f"{incap_od:.2f}%")

# Botón de Descarga PDF
if incap_od > 0 or incap_oi > 0:
    b64_pdf = generar_pdf_base64(incap_od, incap_oi, incap_total_bilateral, modo_evaluacion)
    href = f'<a href="data:application/pdf;base64,{b64_pdf}" download="Dictamen_Pericial_CVC.pdf" style="display: inline-block; padding: 10px 20px; background-color: #4CAF50; color: white; text-align: center; text-decoration: none; border-radius: 5px; font-weight: bold;">📥 Descargar Informe PDF</a>'
    st.markdown(href, unsafe_allow_html=True)
