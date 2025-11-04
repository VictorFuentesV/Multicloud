import os
import io
import json
import zipfile
from pathlib import Path
from typing import List, Tuple, Dict, Any

import re
import fitz  # PyMuPDF
import streamlit as st
import pandas as pd
import numpy as np

from dotenv import load_dotenv

# --- PDF report generation ---
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

# ===========================
# CONFIG GLOBAL
# ===========================
st.set_page_config(page_title="Censura + Análisis Multi-IA", page_icon="🕵️‍♂️🤖", layout="wide")
load_dotenv()

# ===========================
# --- IA LOCAL (Ollama) ----
# ===========================
LLM_MODEL = "llama3.2"
USE_LOCAL_LLM = True  # se puede desactivar en tiempo de ejecución
try:
    import ollama
    _OLLAMA_OK = True
except Exception:
    _OLLAMA_OK = False
    USE_LOCAL_LLM = False

# ------------ Heurísticas / regex ------------
CORP_MARKERS = [
    " LTDA", " LIMITADA", " S.A.", " S A ", " S.A", " E.I.R.L", " EIRL",
    " SPA", " S.P.A", " SOCIEDAD", " COMERCIAL ", " INDUSTRIAL ", " EMPRESA "
]

LABELS = {
    "ASEGURADO": ["ASEGURADO"],
    "RUT": ["RUT", "R.U.T", "RUT:"],
    "TEL": ["TELÉFONO", "TELEFONO", "TEL.", "FONO"],
    "MAIL": ["E-MAIL", "EMAIL", "CORREO", "CORREO ELECTRÓNICO", "CORREO ELECTRONICO"],
}

Y_GAP = 2
Y_BAND = 22
X_PAD_LEFT = 0
X_PAD_RIGHT = 260

RUT_RE   = re.compile(r"\b\d{1,2}\.?\d{3}\.?\d{3}-[0-9Kk]\b")
PHONE_RE = re.compile(r"(?:\+?56)?\s*(?:9\s*)?\d{7,9}")
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[A-Za-z]{2,}")
POLICY_RE= re.compile(r"(?:N[°º]\s*|NRO\s*|NUM\s*|No\.\s*)?(P[ÓO]LIZA|POLIZA)\s*[:\-]?\s*[A-Z0-9\-\.]{5,}", re.IGNORECASE)
CLAIM_RE = re.compile(r"(SINIESTRO|N[°º]\s*DE\s*SINIESTRO)\s*[:\-]?\s*[A-Z0-9\-\.]{4,}", re.IGNORECASE)
PLATE_RE = re.compile(r"\b([A-Z]{2,3}\s*[-·]?\s*\d{2,3}\s*[A-Z0-9]{0,2}|\w{2}\-\w{2}\-\d{2})\b")
VIN_RE   = re.compile(r"\b([A-HJ-NPR-Z0-9]{11,17})\b")
ADDRESS_HINT = re.compile(r"(DIRECCI[ÓO]N|DOMICILIO|CALLE)\b", re.IGNORECASE)

LLM_SYSTEM = (
    "Eres un ayudante que identifica datos sensibles en pólizas de seguros. "
    "Devuelve exclusivamente JSON compacto con claves: "
    '{"names":[], "addresses":[], "policy_numbers":[], "claim_numbers":[], "plates":[], "vins":[], "emails":[], "phones":[], "ids":[]} '
    "No expliques nada. No agregues texto fuera del JSON. "
    "Incluye solo cadenas exactas encontradas en el texto."
)

LLM_USER_TEMPLATE = """Texto de la póliza (solo primera página):
---
{page_text}
---
Instrucciones:
- Extrae nombres de personas (no empresas), direcciones, números de póliza, números de siniestro, patentes/placas, VIN, correos, teléfonos, identificadores como RUT (CL).
- Responde SOLO con JSON válido usando las claves pedidas. Sin comentarios ni explicaciones.
- No inventes: si no estás seguro, deja la lista vacía.
"""

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def words_in_rect(page: fitz.Page, rect: fitz.Rect):
    words = page.get_text("words")
    out = []
    for w in words:
        x0, y0, x1, y1, txt, blk, ln, wn = w
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        if rect.contains(fitz.Point(cx, cy)) and (txt or "").strip():
            out.append(w)
    return out

def union_bbox(words):
    x0 = min(w[0] for w in words)
    y0 = min(w[1] for w in words)
    x1 = max(w[2] for w in words)
    y1 = max(w[3] for w in words)
    return fitz.Rect(x0, y0, x1, y1)

def join_line_text(words_line):
    words_line = sorted(words_line, key=lambda w: (w[5], w[6], w[7]))
    return " ".join((w[4] or "").strip() for w in words_line if (w[4] or "").strip())

def looks_like_company(name_upper: str) -> bool:
    return any(marker in name_upper for marker in CORP_MARKERS)

def redact_value_line(page: fitz.Page, label_rect: fitz.Rect, page_rect: fitz.Rect) -> fitz.Rect | None:
    x0 = max(page_rect.x0, label_rect.x0 - X_PAD_LEFT)
    x1 = min(page_rect.x1, label_rect.x1 + X_PAD_RIGHT)
    y0 = label_rect.y1 + Y_GAP
    y1 = y0 + Y_BAND
    band = fitz.Rect(x0, y0, x1, y1)

    words = words_in_rect(page, band)
    if not words:
        return None

    groups: Dict[Tuple[int,int], list] = {}
    for w in words:
        groups.setdefault((w[5], w[6]), []).append(w)

    best = None
    best_w = -1
    for ws in groups.values():
        r = union_bbox(ws)
        if r.width > best_w:
            best_w = r.width
            best = ws

    if not best:
        return None
    return union_bbox(best)

def search_first_label_rect(page: fitz.Page, variants: list[str]) -> fitz.Rect | None:
    candidates: list[fitz.Rect] = []
    for v in variants:
        hits = page.search_for(v)
        if hits:
            candidates.extend(hits)
    if not candidates:
        return None
    candidates.sort(key=lambda r: (r.y0, r.x0))
    return candidates[0]

def run_local_llm(text: str) -> Dict[str, list]:
    if not USE_LOCAL_LLM or not _OLLAMA_OK:
        return {}
    prompt = LLM_USER_TEMPLATE.format(page_text=text[:6000])
    try:
        resp = ollama.generate(
            model=LLM_MODEL,
            prompt=f"{LLM_SYSTEM}\n\n{prompt}",
            options={"temperature": 0.1}
        )
        raw = resp.get("response", "").strip().strip("` \n")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip(": \n")
        data = json.loads(raw)
        if isinstance(data, dict):
            for k in ["names","addresses","policy_numbers","claim_numbers","plates","vins","emails","phones","ids"]:
                data.setdefault(k, [])
            return data
    except Exception:
        pass
    return {}

def page_words(page: fitz.Page):
    return page.get_text("words")

def match_phrase_rects(page: fitz.Page, phrase: str) -> List[fitz.Rect]:
    phrase_norm = normalize(re.sub(r"[^\w@.\-]+", " ", phrase))
    if not phrase_norm:
        return []
    words = page_words(page)
    norm_words = []
    for w in words:
        x0,y0,x1,y1,txt,blk,ln,wn = w
        t = normalize(re.sub(r"[^\w@.\-]+", " ", txt))
        if t:
            norm_words.append((t, w))
    target_tokens = phrase_norm.split(" ")

    rects = []
    i = 0
    while i < len(norm_words):
        if norm_words[i][0] == target_tokens[0]:
            j = i
            k = 0
            matched = []
            while j < len(norm_words) and k < len(target_tokens):
                if norm_words[j][0] == target_tokens[k]:
                    matched.append(norm_words[j][1])
                    j += 1
                    k += 1
                else:
                    break
            if k == len(target_tokens):
                rects.append(union_bbox(matched))
                i = j
                continue
        i += 1
    return rects

def redact_rects(page: fitz.Page, rects: List[fitz.Rect]):
    for r in rects:
        page.add_redact_annot(r, fill=(0,0,0))

# --------- función de CENSURA (solo primera página) ---------
def redact_pdf_first_page_hybrid(input_bytes: bytes, use_llm: bool = True) -> bytes:
    doc = fitz.open(stream=input_bytes, filetype="pdf")
    if doc.page_count == 0:
        out = doc.write(); doc.close(); return out

    page = doc.load_page(0)
    page_rect = page.rect

    # 1) Etiquetas
    r_label_aseg = search_first_label_rect(page, LABELS["ASEGURADO"])
    if r_label_aseg:
        r_value_aseg = redact_value_line(page, r_label_aseg, page_rect)
        if r_value_aseg:
            words = words_in_rect(page, r_value_aseg)
            candidate = join_line_text(words).strip()
            if candidate and not looks_like_company(f" {candidate.upper()} "):
                page.add_redact_annot(r_value_aseg, fill=(0, 0, 0))

    for key in ("RUT", "TEL", "MAIL"):
        r_lab = search_first_label_rect(page, LABELS[key])
        if r_lab:
            r_val = redact_value_line(page, r_lab, page_rect)
            if r_val:
                page.add_redact_annot(r_val, fill=(0,0,0))

    # 2) Regex en toda la página
    page_text = page.get_text("text")
    for pat in (EMAIL_RE, PHONE_RE, RUT_RE, POLICY_RE, CLAIM_RE, PLATE_RE, VIN_RE):
        for m in pat.finditer(page_text):
            rects = match_phrase_rects(page, m.group(0))
            redact_rects(page, rects)

    # 3) Heurística de direcciones
    for m in ADDRESS_HINT.finditer(page_text):
        rects = match_phrase_rects(page, m.group(0))
        redact_rects(page, rects)

    # 4) IA local (opcional)
    if use_llm and USE_LOCAL_LLM and _OLLAMA_OK:
        llm_data = run_local_llm(page_text) or {}
        candidates = []
        for key in ["names","addresses","policy_numbers","claim_numbers","plates","vins","emails","phones","ids"]:
            vals = llm_data.get(key, []) or []
            if key == "names":
                vals = [v for v in vals if not looks_like_company(f" {v.upper()} ")]
            candidates.extend(vals)
        for c in candidates:
            rects = match_phrase_rects(page, c)
            redact_rects(page, rects)

    # aplicar
    try:
        if page.first_annot:
            page.apply_redactions()
    except Exception:
        page.apply_redactions()

    out = doc.write()
    doc.close()
    return out

# ===========================
# --- IA EN LA NUBE ---------
# ===========================
import google.generativeai as genai
import openai

@st.cache_resource
def get_google_model():
    try:
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        return genai.GenerativeModel("models/gemini-pro-latest")
    except Exception as e:
        st.error(f"Error Google Gemini: {e}")
        return None

@st.cache_resource
def get_openai_client():
    try:
        return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        st.error(f"Error OpenAI: {e}")
        return None

def call_generative_ai(model_choice: str, prompt_text: str) -> str:
    try:
        if model_choice == "Google Gemini":
            model = get_google_model()
            if not model:
                return '{"error": "Gemini no configurado"}'
            resp = model.generate_content(prompt_text)
            return resp.text
        else:
            client = get_openai_client()
            if not client:
                return '{"error": "OpenAI no configurado"}'
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Eres un asistente experto en análisis de seguros. Responde únicamente con un objeto JSON válido sin texto adicional ni explicaciones."},
                    {"role": "user", "content": prompt_text}
                ],
                response_format={"type": "json_object"}
            )
            return resp.choices[0].message.content
    except Exception as e:
        st.error(f"Error en llamada a {model_choice}: {e}")
        return f'{{"error": "Falla API {model_choice}"}}'

# ===========================
# UTILIDADES
# ===========================
def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            return "".join(page.get_text() for page in document)
    except Exception as e:
        st.error(f"No se pudo leer PDF anonimizado: {e}")
        return ""

def highlight_min_deducible_df(df: pd.DataFrame):
    styles = pd.DataFrame('', index=df.index, columns=df.columns)
    cols_to_highlight = [c for c in df.columns if c.startswith('UF ')]
    for col in cols_to_highlight:
        numeric_col = pd.to_numeric(df[col], errors='coerce')
        min_value = numeric_col.min()
        if not pd.isna(min_value):
            styles.loc[numeric_col == min_value, col] = 'background-color: #d4edda; color: #155724; font-weight: bold;'
    return styles

def build_pdf_report(data: dict, nombres_archivos: list[str]) -> bytes:
    """Genera un PDF con: Precios por Deducible, Coberturas comparadas, Resumen y Recomendación."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title="Resumen Comparativo de Pólizas")
    styles = getSampleStyleSheet()
    h1 = styles['Heading1']
    h2 = styles['Heading2']
    p = styles['BodyText']

    elements = []

    # Título principal
    elements.append(Paragraph("Resumen comparativo de pólizas (documentos censurados)", h1))
    elements.append(Spacer(1, 10))

    # 1) Cuadro Resumen — Precios por Deducible
    elements.append(Paragraph("Cuadro Resumen — Precios por Deducible", h2))
    precios = data.get("precios_deducible") or []
    if precios:
        df_precios = pd.DataFrame(precios)
        table_data = [list(df_precios.columns)] + df_precios.fillna("").astype(str).values.tolist()
        t = Table(table_data, repeatRows=1)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F0F0F0')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ]))
        elements.append(t)
    else:
        elements.append(Paragraph("No se encontraron precios por deducible.", p))
    elements.append(Spacer(1, 10))

    # 2) Coberturas comparadas
    elements.append(Paragraph("Coberturas comparadas", h2))
    cob = data.get("coberturas_comparadas") or []
    if cob:
        df_cob = pd.DataFrame(cob)
        colmap = {f"POLIZA_{i+1}": name for i, name in enumerate(nombres_archivos)}
        df_cob.rename(columns=colmap, inplace=True)
        table_data = [list(df_cob.columns)] + df_cob.fillna("").astype(str).values.tolist()
        t2 = Table(table_data, repeatRows=1)
        t2.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#F0F0F0')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ]))
        elements.append(t2)
    else:
        elements.append(Paragraph("No se encontraron coberturas para comparar.", p))
    elements.append(Spacer(1, 10))

    # 3) Resumen y Recomendación
    elements.append(Paragraph("Resumen y Recomendación de la IA", h2))
    ryR = data.get("resumen_y_recomendacion") or {}
    puntos = ryR.get("puntos_fuertes") or []
    if puntos:
        for pt in puntos:
            pol = pt.get('poliza', 'Póliza')
            vent = pt.get('ventaja','')
            elements.append(Paragraph(f"<b>Puntos Fuertes de {pol}:</b> {vent}", p))
    reco = ryR.get("recomendacion")
    if reco:
        elements.append(Spacer(1, 6))
        elements.append(Paragraph(f"<b>Recomendación Final:</b> {reco}", p))

    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes

# ===========================
# UI SIDEBAR
# ===========================
with st.sidebar:
    st.header("⚙️ Configuración")
    model_selection = st.selectbox("Motor de IA en la nube:", ("Google Gemini", "OpenAI GPT-4o"))
    USE_LOCAL_LLM = st.toggle("Usar IA local (Ollama) en la censura", value=USE_LOCAL_LLM and _OLLAMA_OK, help="Si no tienes Ollama, se usa solo censura por reglas/regex.")
    st.caption(f"IA local: {'ACTIVA' if (USE_LOCAL_LLM and _OLLAMA_OK) else 'INACTIVA'} — modelo: {LLM_MODEL}")

# ===========================
# UI PRINCIPAL
# ===========================
st.title("🕵️‍♂️ Censura → 🤖 Análisis Multi-IA de Pólizas")
st.markdown("Sube **2 o más PDFs**. Primero se **anonimizan** (1ª página), luego se **analizan** con IA en la nube para extraer precios por deducible, coberturas y una recomendación.")

# --- Manejo robusto de estado para evitar "reinicios" tras descargar ---
if "uploaded_buffer" not in st.session_state:
    st.session_state.uploaded_buffer = []  # lista de (name, bytes) originales
if "anon_results" not in st.session_state:
    st.session_state.anon_results = []     # lista de (filename, bytes anon)
if "zip_censurados" not in st.session_state:
    st.session_state.zip_censurados = None
if "analysis_data" not in st.session_state:
    st.session_state.analysis_data = None
if "analysis_raw" not in st.session_state:
    st.session_state.analysis_raw = None
if "nombres_archivos" not in st.session_state:
    st.session_state.nombres_archivos = []

uploaded_files = st.file_uploader(
    "Arrastra tus PDFs aquí",
    type=["pdf"],
    accept_multiple_files=True,
    key="uploader_pdfs"
)

# Si llegan archivos nuevos, los persistimos en buffer
if uploaded_files:
    existing = {n for n, _ in st.session_state.uploaded_buffer}
    for f in uploaded_files:
        if f.name not in existing:
            st.session_state.uploaded_buffer.append((f.name, f.read()))

col1, col2 = st.columns([1,1])
with col1:
    if st.button("🛡️🤖  Censurar y analizar", type="primary", key="btn_censurar_analizar"):
        # 1) Censura usando buffer persistente
        if not st.session_state.uploaded_buffer or len(st.session_state.uploaded_buffer) == 0:
            st.warning("Sube al menos 2 archivos para continuar.")
        else:
            st.session_state.anon_results = []
            st.session_state.analysis_data = None
            st.session_state.analysis_raw = None
            st.session_state.nombres_archivos = []

            prog = st.progress(0.0, text="Censurando…")
            total = len(st.session_state.uploaded_buffer)
            for i, (name, raw) in enumerate(st.session_state.uploaded_buffer, start=1):
                try:
                    out_bytes = redact_pdf_first_page_hybrid(raw, use_llm=USE_LOCAL_LLM)
                    st.session_state.anon_results.append((name, out_bytes))
                except Exception as e:
                    st.error(f"Error al censurar {name}: {e}")
                prog.progress(i/total, text=f"Censurado {i}/{total}")
            prog.empty()

            if st.session_state.anon_results:
                st.success(f"Censura completa: {len(st.session_state.anon_results)} archivo(s).")
                # ZIP persistente
                mem = io.BytesIO()
                with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                    for name, bts in st.session_state.anon_results:
                        zf.writestr(f"anon_{Path(name).stem}.pdf", bts)
                mem.seek(0)
                st.session_state.zip_censurados = mem.read()

                # 2) Análisis Cloud inmediato
                with st.spinner(f"Analizando documentos con {model_selection}…"):
                    textos_polizas = []
                    nombres_archivos = []
                    for i, (name, bts) in enumerate(st.session_state.anon_results, start=1):
                        txt = extract_text_from_pdf_bytes(bts)
                        if txt:
                            textos_polizas.append(
                                f"""--- INICIO POLIZA_{i} (anon_{Path(name).stem}.pdf) ---
{txt}
--- FIN POLIZA_{i} (anon_{Path(name).stem}.pdf) ---"""
                            )
                            nombres_archivos.append(f"anon_{Path(name).stem}.pdf")
                    st.session_state.nombres_archivos = nombres_archivos

                    prompt_parts = [
                        "Eres un asistente experto para una corredora de seguros. Tu tarea es analizar el texto de las siguientes pólizas censuradas y extraer la información en un formato JSON estricto. No incluyas texto adicional antes o después del JSON.",
                        "\n\n".join(textos_polizas),
                        "\n\nAnaliza las pólizas y genera un objeto JSON con la siguiente estructura. Sigue estas reglas rigurosamente:",
                        """
                        {
                          "precios_deducible": [
                            {
                              "compañia": "Nombre de la Aseguradora (ej. 'HDI Seguros')",
                              "tipo_pago": "Método de pago encontrado (ej. 'PAT', 'PAC', 'Contado / PAC / PAT', 'Sin datos')",
                              "UF 0": "valor_numerico O null",
                              "UF 3": "valor_numerico O null",
                              "UF 5": "valor_numerico O null",
                              "UF 10": "valor_numerico O null",
                              "UF 15": "valor_numerico O null",
                              "UF 20": "valor_numerico O null",
                              "UF 30": "valor_numerico O null"
                            }
                          ],
                          "coberturas_comparadas": [
                            {
                              "Cobertura": "Nombre de la Cobertura (ej. 'actos maliciosos')",
                              "POLIZA_1": "Detalle para la póliza 1 (ej. 'Sí', 'UF 100', 'No aplica')",
                              "POLIZA_2": "Detalle para la póliza 2"
                            }
                          ],
                          "resumen_y_recomendacion": {
                            "puntos_fuertes": [
                              {"poliza": "Nombre Archivo 1", "ventaja": "Ventaja principal de esta póliza."},
                              {"poliza": "Nombre Archivo 2", "ventaja": "Ventaja principal de esta póliza."}
                            ],
                            "recomendacion": "Recomienda una póliza y explica por qué de forma concisa, considerando la relación entre coberturas y precios de deducibles."
                          }
                        }
                        """,
                        "INSTRUCCIONES CRÍTICAS:",
                        "1. Para 'precios_deducible': Extrae el valor de la prima total ANUAL o el total de las cuotas en UF. Prioriza PAC/PAT; si no hay, usa Contado/Aviso de Cobranza. Valores faltantes deben ser null.",
                        "2. Para 'coberturas_comparadas': Las claves deben ser 'Cobertura' y 'POLIZA_1', 'POLIZA_2', etc., según el orden.",
                        "3. GENERAL: Sé extremadamente preciso con los datos numéricos."
                    ]
                    final_prompt = "\n".join(prompt_parts)

                    try:
                        response_text = call_generative_ai(model_selection, final_prompt)
                        cleaned = (response_text or "").strip().replace("```json", "").replace("```", "").replace("`", "")
                        st.session_state.analysis_raw = cleaned
                        data = json.loads(cleaned)
                        st.session_state.analysis_data = data
                        st.success("¡Análisis completado!")
                    except json.JSONDecodeError:
                        st.error("La IA no devolvió JSON válido. Respuesta recibida:")
                        st.code(st.session_state.analysis_raw or "(vacío)")
                    except Exception as e:
                        st.error(f"Error inesperado en el análisis: {e}")

with col2:
    # Descarga persistente del ZIP (si existe), sin afectar estado ni resultados
    if st.session_state.zip_censurados:
        st.download_button(
            "⬇️ Descargar ZIP de PDFs censurados",
            data=st.session_state.zip_censurados,
            file_name="pdfs_censurados.zip",
            mime="application/zip",
            key="download_zip_censurados"
        )

# Vista previa de la 1ª página censurada
if st.session_state.anon_results:
    st.subheader("🔎 Vistas previas (página 1)")
    pv_cols = st.columns(2)
    for idx, (name, bts) in enumerate(st.session_state.anon_results):
        with pv_cols[idx % 2]:
            try:
                with fitz.open(stream=bts, filetype="pdf") as doc:
                    page = doc.load_page(0)
                    img = page.get_pixmap(matrix=fitz.Matrix(1, 1)).tobytes("png")
                    st.image(img, caption=f"anon_{Path(name).stem}.pdf — pág. 1")
            except Exception as e:
                st.warning(f"No se pudo previsualizar {name}: {e}")
else:
    st.info("Sube y censura tus PDFs para ver aquí las vistas previas y habilitar el análisis.")

st.markdown("---")

# Paso 2 (UNIFICADO): Mostrar resultados del análisis si existen en estado
if st.session_state.get("analysis_data"):
    data = st.session_state.analysis_data
    nombres_archivos = st.session_state.get("nombres_archivos", [])

    st.header("📈 Cuadro Resumen — Precios por Deducible")
    if data.get("precios_deducible"):
        df_precios = pd.DataFrame(data["precios_deducible"])
        st.dataframe(
            df_precios.style.apply(highlight_min_deducible_df, axis=None).format(precision=4, na_rep="None"),
            use_container_width=True
        )

    st.header("📝 Coberturas comparadas")
    if data.get("coberturas_comparadas"):
        df_cob = pd.DataFrame(data["coberturas_comparadas"])
        colmap = {f"POLIZA_{i+1}": name for i, name in enumerate(nombres_archivos)}
        df_cob.rename(columns=colmap, inplace=True)
        st.dataframe(df_cob, use_container_width=True, hide_index=True)

    st.header("🧠 Resumen y Recomendación de la IA")
    if data.get("resumen_y_recomendacion"):
        resumen = data["resumen_y_recomendacion"]
        for punto in resumen.get("puntos_fuertes", []) or []:
            st.markdown(f"**Puntos Fuertes de `{punto.get('poliza','N/A')}`:** {punto.get('ventaja','N/A')}")
        if resumen.get("recomendacion"):
            st.info(f"**Recomendación Final:** {resumen['recomendacion']}")

    # --- Botón para descargar PDF consolidado ---
    try:
        pdf_bytes = build_pdf_report(data, nombres_archivos)
        st.download_button(
            label="📄 Descargar informe en PDF",
            data=pdf_bytes,
            file_name="informe_comparativo_polizas.pdf",
            mime="application/pdf",
            key="download_pdf_informe"
        )
    except Exception as e:
        st.warning(f"No se pudo generar el PDF del informe: {e}")

