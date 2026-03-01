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
            os.remove(tmp_od.name)
        if img_oi_orig is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_oi:
                cv2.imwrite(tmp_oi.name, img_oi_orig)
                # Reducido a 75mm de ancho
                pdf.image(tmp_oi.name, x=110, y=y_images, w=75) 
            os.remove(tmp_oi.name)
        # Salto vertical gigante (100mm)
        pdf.set_y(y_images + 100) 
    else:
        img_val = img_od_orig if img_od_orig is not None else img_oi_orig
        if img_val is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cv2.imwrite(tmp.name, img_val)
                # Una sola imagen, 90mm de ancho y centrada
                pdf.image(tmp.name, x=60, y=y_images, w=90) 
            os.remove(tmp.name)
        # Salto vertical gigante (120mm)
        pdf.set_y(y_images + 120)

    # 3. RESULTADOS MATEMÁTICOS (Totalmente a salvo de solapamientos)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "RESULTADOS DE LA EVALUACION (AREA 40 GRADOS)", 0, 1, 'L')
    pdf.ln(2)
    
    if incap_od > 0:
        pdf.set_fill_color(235, 245, 255) 
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, " OJO DERECHO (OD)", 0, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"   - Grados de perdida visual:  {grados_od:.1f} grados", 0, 1)
        pdf.cell(0, 8, f"   - Incapacidad Unilateral:    {incap_od:.2f}%", 0, 1)
        pdf.ln(3)
        
    if incap_oi > 0:
        pdf.set_fill_color(235, 245, 255)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, " OJO IZQUIERDO (OI)", 0, 1, 'L', fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f"   - Grados de perdida visual:  {grados_oi:.1f} grados", 0, 1)
        pdf.cell(0, 8, f"   - Incapacidad Unilateral:    {incap_oi:.2f}%", 0, 1)
        pdf.ln(3)
        
    # 4. INCAPACIDAD TOTAL
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 14)
    if modo == "Bilateral (OD y OI)":
        pdf.set_fill_color(46, 134, 193) 
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 14, f" INCAPACIDAD TOTAL BILATERAL: {incap_total:.2f}%", 0, 1, 'C', fill=True)
    else:
        pdf.set_fill_color(46, 134, 193)
        pdf.set_text_color(255, 255, 255)
        val = incap_od if incap_od > 0 else incap_oi
        pdf.cell(0, 14, f" INCAPACIDAD UNILATERAL DEFINITIVA: {val:.2f}%", 0, 1, 'C', fill=True)
        
    # 5. FIRMA
    pdf.set_text_color(0, 0, 0)
    pdf.ln(15)
    pdf.set_draw_color(0, 0, 0)
    pdf.line(65, pdf.get_y(), 145, pdf.get_y())
    pdf.set_font("Arial", '', 10)
    pdf.cell(0, 5, "Firma y Sello del Perito Medico", 0, 1, 'C')
    
    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return base64.b64encode(pdf_bytes).decode()

# ==========================================
# INTERFAZ WEB (`app.py`)
# ==========================================

modo_evaluacion = st.radio("Seleccione el Tipo de Evaluación:", ["Unilateral (1 Ojo)", "Bilateral (OD y OI)"], horizontal=True)
st.divider()

def procesar_panel_ojo(titulo_ojo, key_suffix):
    archivo = st.file_uploader(f"Subir estudio - {titulo_ojo}", type=["jpg", "jpeg", "png"], key=f"file_{key_suffix}")
    incapacidad_final, grados_finales, img_original = 0.0, 0.0, None
    
    if archivo is not None:
        with st.spinner(f"Escaneando {titulo_ojo}..."):
            nparr = np.frombuffer(archivo.getvalue(), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # GUARDAMOS LA IMAGEN ORIGINAL INTACTA PARA EL PDF
            img_original = img.copy()
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
            
            centro, borrador_anti_regla, dist_60 = find_and_clean_axes(thresh)
            pixels_por_10_grados = float(dist_60 / 6.0)
            
            img_auditoria_bin, t_cuad, t_circ = detect_and_classify_symbols(thresh, borrador_anti_regla, centro, pixels_por_10_grados)
            
            # IMAGEN CON CAJITAS SOLO PARA MOSTRAR EN PANTALLA DE TRABAJO
            img_pantalla = img.copy()
            for i in range(3):
                mask = img_auditoria_bin[:,:,i] != 255
                img_pantalla[mask, i] = img_auditoria_bin[mask, i]
            cv2.circle(img_pantalla, centro, int(4.0 * pixels_por_10_grados), (0, 165, 255), 3)

            st.image(Image.fromarray(cv2.cvtColor(img_pantalla, cv2.COLOR_BGR2RGB)), caption=f"Auditoría Visual {titulo_ojo}", use_container_width=True)
            
            st.markdown(f"**Corrección Pericial**")
            col_a, col_b = st.columns(2)
            with col_a:
                cuadrados_final = st.number_input("Cuadrados (Fallados):", min_value=0, max_value=104, value=t_cuad, step=1, key=f"cuad_{key_suffix}")
            with col_b:
                circulos_final = st.number_input("Círculos (Vistos):", min_value=0, max_value=104, value=t_circ, step=1, key=f"circ_{key_suffix}")
                
            grados_finales = (cuadrados_final / 104.0) * 320.0
            incapacidad_final = (grados_finales / 320.0) * 100 * 0.25
            
            st.metric(f"Incapacidad {titulo_ojo}", f"{incapacidad_final:.2f}%")
            
    return incapacidad_final, grados_finales, img_original

# Layout
if modo_evaluacion == "Unilateral (1 Ojo)":
    incap_od, grados_od, img_od_orig = procesar_panel_ojo("Ojo Evaluado", "unico")
    incap_oi, grados_oi, img_oi_orig = 0.0, 0.0, None
else:
    col_izq, col_der = st.columns(2)
    with col_izq:
        incap_od, grados_od, img_od_orig = procesar_panel_ojo("Ojo Derecho (OD)", "od")
    with col_der:
        incap_oi, grados_oi, img_oi_orig = procesar_panel_ojo("Ojo Izquierdo (OI)", "oi")

st.divider()

# ==========================================
# DICTAMEN FINAL Y PDF
# ==========================================
st.header("📋 Dictamen Legal y Exportación")

# Campo solo para nombrar el archivo al descargar
nombre_archivo_input = st.text_input("Nombre para guardar el archivo PDF (opcional):", placeholder="Ej: Perez_Juan")

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
    if incap_od > 0:
        st.metric("INCAPACIDAD UNILATERAL", f"{incap_od:.2f}%")

if incap_od > 0 or incap_oi > 0:
    
    nombre_archivo = nombre_archivo_input.strip().replace(" ", "_") if nombre_archivo_input.strip() else "Dictamen"
        
    b64_pdf = generar_pdf_moderno(incap_od, grados_od, img_od_orig, incap_oi, grados_oi, img_oi_orig, incap_total_bilateral, modo_evaluacion)
    
    html_btn = f'''
    <a href="data:application/pdf;base64,{b64_pdf}" download="Dictamen_Pericial_{nombre_archivo}.pdf" style="display: block; padding: 15px; background-color: #2980b9; color: white; text-align: center; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 18px; margin-top: 20px;">
        📥 DESCARGAR INFORME PERICIAL PDF
    </a>
    '''
    st.markdown(html_btn, unsafe_allow_html=True)
