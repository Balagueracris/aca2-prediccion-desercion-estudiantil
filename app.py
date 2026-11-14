import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Predicción de Deserción Estudiantil",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "modelo_mlp_desercion.keras"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocesador.joblib"
METADATA_PATH = BASE_DIR / "models" / "metadata.joblib"
DATA_PATH = BASE_DIR / "data" / "data.csv"


# ============================================================
# 2. ESTILOS VISUALES
# ============================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    :root {
        --cun-navy: #101B45;
        --cun-blue: #164E91;
        --cun-purple: #55208C;
        --cun-magenta: #8B2E91;
        --cun-cyan: #24A6C7;
        --cun-light: #E9E7F5;
    }

    .stApp {
        background:
            radial-gradient(circle at 8% 4%, rgba(36, 166, 199, .18), transparent 27%),
            radial-gradient(circle at 90% 8%, rgba(139, 46, 145, .22), transparent 30%),
            linear-gradient(145deg, #09152f 0%, #111b42 48%, #21123f 100%);
        color: #F7F5FF;
    }

    [data-testid="stHeader"] {
        background: rgba(9, 21, 47, .72);
    }

    .block-container {
        max-width: 1240px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 2.2rem 2.3rem;
        border: 1px solid rgba(220, 210, 255, .23);
        border-radius: 24px;
        background:
            linear-gradient(115deg, rgba(17, 49, 105, .97) 0%,
            rgba(57, 31, 112, .96) 58%, rgba(112, 39, 119, .92) 100%);
        box-shadow: 0 18px 55px rgba(0,0,0,.28);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }

    .hero:after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        right: -95px;
        top: -145px;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(255,255,255,.035),
                    0 0 0 58px rgba(255,255,255,.025);
    }

    .hero-kicker {
        color: #B9EAF5;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .14em;
        text-transform: uppercase;
        margin-bottom: .7rem;
    }

    .hero h1 {
        color: white;
        font-size: clamp(1.8rem, 4vw, 2.65rem);
        font-weight: 800;
        line-height: 1.12;
        margin: 0 0 .65rem 0;
    }

    .hero p {
        color: #E6E2F8;
        font-size: 1rem;
        max-width: 760px;
        margin: 0;
        line-height: 1.65;
    }

    .section-heading {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F7F5FF;
        margin: .3rem 0 .25rem 0;
    }

    .section-subtitle {
        color: #C0C4DD;
        font-size: .94rem;
        margin-bottom: 1rem;
    }

    .info-card {
        background: linear-gradient(145deg, rgba(27, 48, 94, .88), rgba(65, 35, 100, .78));
        border: 1px solid rgba(204, 190, 255, .22);
        border-radius: 16px;
        padding: 1rem 1.1rem;
        min-height: 105px;
        box-shadow: 0 8px 24px rgba(0,0,0,.12);
    }

    .info-card .label {
        color: #C4C9E2;
        font-size: .8rem;
        font-weight: 700;
        margin-bottom: .35rem;
    }

    .info-card .value {
        color: #FFFFFF;
        font-size: 1.2rem;
        font-weight: 800;
    }

    div[data-testid="stForm"] {
        background: rgba(13, 25, 58, .78);
        border: 1px solid rgba(190, 179, 239, .24);
        border-radius: 20px;
        padding: 1.25rem 1.35rem 1.1rem 1.35rem;
        box-shadow: 0 12px 34px rgba(0,0,0,.16);
    }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stNumberInput"] label {
        color: #F0EDFF;
        font-weight: 600;
        font-size: .9rem;
    }

    /* Todos los campos mantienen el mismo fondo oscuro */
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="input"] input,
    input[type="number"],
    input[type="text"] {
        border-radius: 10px !important;
        background-color: #302D2A !important;
        color: #F7F5FF !important;
        -webkit-text-fill-color: #F7F5FF !important;
        border: 1px solid rgba(190, 179, 239, .28) !important;
        box-shadow: none !important;
    }

    /* Evita que el campo numérico vuelva a blanco al enfocarlo */
    div[data-testid="stNumberInput"] input:focus,
    div[data-baseweb="input"] input:focus,
    input[type="number"]:focus {
        background-color: #302D2A !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border: 1px solid #8B2E91 !important;
        box-shadow: 0 0 0 1px #8B2E91 !important;
    }

    /* Fondo oscuro para las opciones desplegables */
    div[role="listbox"],
    ul[role="listbox"],
    li[role="option"] {
        background-color: #25232A !important;
        color: #F7F5FF !important;
    }

    li[role="option"]:hover,
    li[role="option"][aria-selected="true"] {
        background-color: #49305F !important;
        color: #FFFFFF !important;
    }

    /* Controles + y - de los campos numéricos */
    div[data-testid="stNumberInput"] button {
        background-color: #302D2A !important;
        color: #F7F5FF !important;
        border-color: rgba(190, 179, 239, .25) !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        border: 0;
        border-radius: 12px;
        padding: .78rem 1.2rem;
        font-weight: 800;
        background: linear-gradient(100deg, #1769AA, #652A91 65%, #982F88);
        color: white;
        box-shadow: 0 8px 22px rgba(85, 32, 140, .28);
        transition: transform .15s ease, box-shadow .15s ease;
    }

    div[data-testid="stFormSubmitButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 28px rgba(36, 166, 199, .25);
        color: white;
        border: 0;
    }

    .result-card {
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        border: 1px solid rgba(204, 190, 255, .24);
        background: linear-gradient(115deg, rgba(24, 54, 105, .9), rgba(79, 37, 112, .88));
        margin: .8rem 0 1rem 0;
    }

    .result-title {
        color: #D1D5EE;
        font-size: .82rem;
        font-weight: 800;
        letter-spacing: .08em;
        text-transform: uppercase;
    }

    .result-value {
        color: #FFFFFF;
        font-size: 1.65rem;
        font-weight: 800;
        margin-top: .3rem;
    }

    .footnote {
        color: #B4B8D2;
        font-size: .82rem;
        text-align: center;
        padding-top: 1rem;
    }

    [data-testid="stAlert"] {
        border-radius: 14px;
    }

    h1, h2, h3, h4 {
        color: #F7F5FF;
    }

    hr {
        border-color: rgba(190, 179, 239, .2);
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. TRADUCCIONES DE VARIABLES
# ============================================================

LABELS = {
    "Marital status": "Estado civil",
    "Application mode": "Modalidad de ingreso",
    "Application order": "Orden de solicitud",
    "Course": "Programa académico",
    "Daytime/evening attendance": "Jornada académica",
    "Previous qualification": "Nivel de estudios previo",
    "Previous qualification (grade)": "Calificación de estudios previos",
    "Nacionality": "Nacionalidad",
    "Mother's qualification": "Nivel educativo de la madre",
    "Father's qualification": "Nivel educativo del padre",
    "Mother's occupation": "Ocupación de la madre",
    "Father's occupation": "Ocupación del padre",
    "Admission grade": "Nota de admisión",
    "Displaced": "Condición de desplazamiento",
    "Educational special needs": "Necesidades educativas especiales",
    "Debtor": "Tiene deudas pendientes",
    "Tuition fees up to date": "Matrícula al día",
    "Gender": "Género",
    "Scholarship holder": "Beneficiario de beca",
    "Age at enrollment": "Edad al momento de la matrícula",
    "International": "Estudiante internacional",
    "Curricular units 1st sem (credited)": "Asignaturas homologadas del primer semestre",
    "Curricular units 1st sem (enrolled)": "Asignaturas inscritas del primer semestre",
    "Curricular units 1st sem (evaluations)": "Evaluaciones del primer semestre",
    "Curricular units 1st sem (approved)": "Asignaturas aprobadas del primer semestre",
    "Curricular units 1st sem (grade)": "Promedio del primer semestre",
    "Curricular units 1st sem (without evaluations)": "Asignaturas sin evaluación del primer semestre",
    "Unemployment rate": "Tasa de desempleo",
    "Inflation rate": "Tasa de inflación",
    "GDP": "Producto interno bruto (PIB)"
}

CLASS_LABELS = {
    "Dropout": "Deserción estudiantil",
    "Enrolled": "Continúa matriculado",
    "Graduate": "Graduación"
}

# Las etiquetas se simplifican para el usuario, pero cada código
# conserva su valor original al enviarse al preprocesador.
CATEGORY_LABELS = {
    "Marital status": {
        1: "Soltero/a", 2: "Casado/a", 3: "Viudo/a",
        4: "Divorciado/a", 5: "Unión de hecho",
        6: "Separado/a legalmente"
    },
    "Application mode": {
        1: "Convocatoria general — primera fase",
        2: "Admisión por normativa especial (612/93)",
        5: "Cupo especial de Azores",
        7: "Ingreso con otros estudios superiores",
        10: "Admisión por normativa especial (854-B/99)",
        15: "Estudiante internacional de pregrado",
        16: "Cupo especial de Madeira",
        17: "Convocatoria general — segunda fase",
        18: "Convocatoria general — tercera fase",
        26: "Admisión por plan de estudios diferente",
        27: "Ingreso desde otra institución",
        39: "Ingreso para mayores de 23 años",
        42: "Traslado universitario",
        43: "Cambio de carrera",
        44: "Ingreso con diploma de especialización tecnológica",
        51: "Cambio de institución o carrera",
        53: "Ingreso con diploma de ciclo corto",
        57: "Cambio de institución o carrera — internacional"
    },
    "Course": {
        33: "Tecnologías de producción de biocombustibles",
        171: "Diseño de animación y multimedia",
        8014: "Trabajo social — jornada nocturna",
        9003: "Agronomía",
        9070: "Diseño de comunicación",
        9085: "Enfermería veterinaria",
        9119: "Ingeniería informática",
        9130: "Equinocultura",
        9147: "Gestión",
        9238: "Trabajo social",
        9254: "Turismo",
        9500: "Enfermería",
        9556: "Higiene oral",
        9670: "Gestión de publicidad y marketing",
        9773: "Periodismo y comunicación",
        9853: "Educación básica",
        9991: "Gestión — jornada nocturna"
    },
    "Daytime/evening attendance": {
        0: "Jornada nocturna", 1: "Jornada diurna"
    },
    "Previous qualification": {
        1: "Educación secundaria",
        2: "Licenciatura de educación superior",
        3: "Título de educación superior",
        4: "Maestría",
        5: "Doctorado",
        6: "Estudios superiores incompletos",
        9: "12.º año sin finalizar",
        10: "11.º año sin finalizar",
        12: "Otro — 11.º año de escolaridad",
        14: "10.º año de escolaridad",
        15: "10.º año sin finalizar",
        19: "Educación básica — tercer ciclo",
        38: "Educación básica — segundo ciclo",
        39: "Especialización tecnológica",
        40: "Título de educación superior — primer ciclo",
        42: "Curso técnico superior profesional",
        43: "Maestría — segundo ciclo"
    },
    "Nacionality": {
        1: "Portuguesa", 2: "Alemana", 6: "Española",
        11: "Italiana", 13: "Neerlandesa", 14: "Inglesa",
        17: "Lituana", 21: "Angoleña", 22: "Caboverdiana",
        24: "Guineana", 25: "Mozambiqueña", 26: "Santomense",
        32: "Turca", 41: "Brasileña", 62: "Rumana",
        100: "Moldava", 101: "Mexicana", 103: "Ucraniana",
        105: "Rusa", 108: "Cubana", 109: "Colombiana"
    },
    "Mother's qualification": {
        1: "Educación secundaria — 12.º año",
        2: "Licenciatura de educación superior",
        3: "Título de educación superior",
        4: "Maestría", 5: "Doctorado",
        6: "Estudios superiores incompletos",
        9: "12.º año sin finalizar", 10: "11.º año sin finalizar",
        11: "7.º año de escolaridad", 12: "Otro — 11.º año",
        14: "10.º año de escolaridad", 18: "Comercio general",
        19: "Educación básica — tercer ciclo",
        22: "Curso técnico profesional",
        26: "7.º año de escolaridad",
        27: "Segundo ciclo de secundaria general",
        29: "9.º año sin finalizar", 30: "8.º año de escolaridad",
        34: "Desconocido", 35: "No sabe leer ni escribir",
        36: "Lee, pero no completó el 4.º año",
        37: "Educación básica — primer ciclo",
        38: "Educación básica — segundo ciclo",
        39: "Especialización tecnológica",
        40: "Título de educación superior",
        41: "Estudios superiores especializados",
        42: "Curso técnico superior profesional",
        43: "Maestría", 44: "Doctorado"
    },
    "Father's qualification": {
        1: "Educación secundaria — 12.º año",
        2: "Licenciatura de educación superior",
        3: "Título de educación superior",
        4: "Maestría", 5: "Doctorado",
        6: "Estudios superiores incompletos",
        9: "12.º año sin finalizar", 10: "11.º año sin finalizar",
        11: "7.º año de escolaridad", 12: "Otro — 11.º año",
        13: "Segundo año de secundaria complementaria",
        14: "10.º año de escolaridad",
        18: "Comercio general",
        19: "Educación básica — tercer ciclo",
        20: "Secundaria complementaria",
        22: "Curso técnico profesional",
        25: "Secundaria complementaria sin finalizar",
        26: "7.º año de escolaridad",
        27: "Segundo ciclo de secundaria general",
        29: "9.º año sin finalizar", 30: "8.º año de escolaridad",
        31: "Administración y comercio",
        33: "Contabilidad y administración",
        34: "Desconocido", 35: "No sabe leer ni escribir",
        36: "Lee, pero no completó el 4.º año",
        37: "Educación básica — primer ciclo",
        38: "Educación básica — segundo ciclo",
        39: "Especialización tecnológica",
        40: "Título de educación superior",
        41: "Estudios superiores especializados",
        42: "Curso técnico superior profesional",
        43: "Maestría", 44: "Doctorado"
    },
    "Mother's occupation": {
        0: "Estudiante",
        1: "Dirección y administración",
        2: "Profesionales científicos e intelectuales",
        3: "Técnicos y profesionales de nivel medio",
        4: "Personal administrativo",
        5: "Servicios, seguridad y ventas",
        6: "Agricultura, pesca y silvicultura",
        7: "Industria, construcción y artesanía",
        8: "Operación de maquinaria",
        9: "Ocupaciones no cualificadas",
        10: "Fuerzas armadas",
        90: "Otra situación", 99: "Sin información",
        122: "Profesionales de la salud", 123: "Docentes",
        125: "Especialistas en tecnologías de información",
        131: "Técnicos de ciencias e ingeniería",
        132: "Técnicos de salud",
        134: "Técnicos de servicios sociales y culturales",
        141: "Empleados de oficina",
        143: "Operadores contables y financieros",
        144: "Apoyo administrativo",
        151: "Servicios personales", 152: "Ventas",
        153: "Cuidado personal",
        171: "Construcción", 173: "Artesanía e impresión",
        175: "Industria y manufactura",
        191: "Limpieza", 192: "Agricultura no cualificada",
        193: "Industria y transporte no cualificados",
        194: "Preparación de alimentos"
    },
    "Father's occupation": {
        0: "Estudiante",
        1: "Dirección y administración",
        2: "Profesionales científicos e intelectuales",
        3: "Técnicos y profesionales de nivel medio",
        4: "Personal administrativo",
        5: "Servicios, seguridad y ventas",
        6: "Agricultura, pesca y silvicultura",
        7: "Industria, construcción y artesanía",
        8: "Operación de maquinaria",
        9: "Ocupaciones no cualificadas",
        10: "Fuerzas armadas",
        90: "Otra situación", 99: "Sin información",
        101: "Oficiales de las fuerzas armadas",
        102: "Sargentos de las fuerzas armadas",
        103: "Otro personal militar",
        112: "Dirección administrativa y comercial",
        114: "Dirección de hotelería y comercio",
        121: "Profesionales de ciencias e ingeniería",
        122: "Profesionales de la salud",
        123: "Docentes",
        124: "Finanzas y administración",
        131: "Técnicos de ciencias e ingeniería",
        132: "Técnicos de salud",
        134: "Servicios sociales y culturales",
        135: "Tecnologías de información",
        141: "Empleados de oficina",
        143: "Operadores contables y financieros",
        144: "Apoyo administrativo",
        151: "Servicios personales", 152: "Ventas",
        153: "Cuidado personal", 154: "Seguridad",
        161: "Agricultura y ganadería",
        163: "Agricultura y pesca de subsistencia",
        171: "Construcción", 172: "Metalurgia",
        174: "Electricidad y electrónica",
        175: "Industria y manufactura",
        181: "Operación de instalaciones y maquinaria",
        182: "Ensamblaje",
        183: "Transporte y conducción",
        192: "Agricultura no cualificada",
        193: "Industria y transporte no cualificados",
        194: "Preparación de alimentos",
        195: "Ventas ambulantes y servicios en la calle"
    }
}

BINARY_LABELS = {
    "Displaced": {0: "No", 1: "Sí"},
    "Educational special needs": {0: "No", 1: "Sí"},
    "Debtor": {0: "No", 1: "Sí"},
    "Tuition fees up to date": {0: "No", 1: "Sí"},
    "Gender": {0: "Femenino", 1: "Masculino"},
    "Scholarship holder": {0: "No", 1: "Sí"},
    "International": {0: "No", 1: "Sí"}
}


def etiqueta_categoria(columna, valor):
    codigo = int(valor)
    if columna in BINARY_LABELS:
        return BINARY_LABELS[columna].get(codigo, f"Código {codigo}")
    return CATEGORY_LABELS.get(columna, {}).get(codigo, f"Código {codigo}")


# ============================================================
# 4. CARGA DE ARTEFACTOS
# ============================================================

@st.cache_resource
def cargar_artefactos():
    modelo = tf.keras.models.load_model(MODEL_PATH)
    preprocesador = joblib.load(PREPROCESSOR_PATH)
    metadatos = joblib.load(METADATA_PATH)

    dataset = pd.read_csv(DATA_PATH, sep=";")
    dataset.columns = dataset.columns.str.strip()

    return modelo, preprocesador, metadatos, dataset


try:
    modelo, preprocesador, metadatos, dataset = cargar_artefactos()
except Exception as error:
    st.error("No fue posible cargar los archivos del modelo. Revisa las carpetas data y models.")
    st.exception(error)
    st.stop()


# ============================================================
# 5. ENCABEZADO VISUAL
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">Analítica académica · Aprendizaje automático</div>
    <h1>Predicción de deserción estudiantil</h1>
    <p>
        Una herramienta de apoyo para identificar posibles situaciones de
        riesgo académico a partir de información de ingreso y desempeño
        del primer semestre.
    </p>
</div>
""", unsafe_allow_html=True)

columnas_modelo = metadatos["feature_columns"]
variables_categoricas = metadatos["categorical_features"]
variables_numericas = metadatos["numeric_features"]
clases = metadatos["class_names"]

m1, m2, m3 = st.columns(3)
with m1:
    st.markdown("""
    <div class="info-card">
        <div class="label">MODELO PREDICTIVO</div>
        <div class="value">Red neuronal MLP</div>
    </div>
    """, unsafe_allow_html=True)
with m2:
    st.markdown(f"""
    <div class="info-card">
        <div class="label">VARIABLES ANALIZADAS</div>
        <div class="value">{len(columnas_modelo)} características</div>
    </div>
    """, unsafe_allow_html=True)
with m3:
    st.markdown("""
    <div class="info-card">
        <div class="label">ENFOQUE</div>
        <div class="value">Prevención temprana</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.info(
    "El análisis utiliza información disponible al finalizar el primer semestre. "
    "Las variables del segundo semestre fueron excluidas para evitar utilizar "
    "información futura."
)

# ============================================================
# 6. FORMULARIO
# ============================================================

st.markdown('<div class="section-heading">📝 Datos del estudiante</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Completa la información para estimar el resultado académico.</div>',
    unsafe_allow_html=True
)

valores_entrada = {}

with st.form("formulario_prediccion"):
    col_izq, col_der = st.columns(2)

    for indice, columna in enumerate(columnas_modelo):
        contenedor = col_izq if indice % 2 == 0 else col_der
        etiqueta = LABELS.get(columna, columna)

        with contenedor:
            # Nacionalidad se presenta con nombres intuitivos, aunque el modelo
            # la recibe como variable numérica: se conserva su código original.
            if columna in variables_categoricas or columna == "Nacionality":
                opciones = sorted(dataset[columna].dropna().unique().tolist())
                valor = st.selectbox(
                    etiqueta,
                    options=opciones,
                    format_func=lambda x, c=columna: etiqueta_categoria(c, x),
                    key=f"campo_{columna}"
                )
                valores_entrada[columna] = int(valor) if columna == "Nacionality" else valor

            elif columna in variables_numericas:
                serie = dataset[columna].dropna()
                minimo = float(serie.min())
                maximo = float(serie.max())
                mediana = float(serie.median())

                if serie.nunique() <= 2:
                    opciones = sorted(serie.unique().tolist())
                    traducciones = BINARY_LABELS.get(columna, {})
                    valor = st.selectbox(
                        etiqueta,
                        options=opciones,
                        format_func=lambda x, t=traducciones: t.get(
                            int(x), f"Código {int(x)}"
                        ),
                        key=f"campo_{columna}"
                    )
                else:
                    es_entero = np.allclose(
                        serie.to_numpy(dtype=float),
                        np.round(serie.to_numpy(dtype=float))
                    )

                    if es_entero:
                        valor = st.number_input(
                            etiqueta,
                            min_value=int(round(minimo)),
                            max_value=int(round(maximo)),
                            value=int(round(mediana)),
                            step=1,
                            key=f"campo_{columna}"
                        )
                        valor = int(valor)
                    else:
                        valor = st.number_input(
                            etiqueta,
                            min_value=float(minimo),
                            max_value=float(maximo),
                            value=float(mediana),
                            step=0.1,
                            format="%.2f",
                            key=f"campo_{columna}"
                        )
                        valor = float(valor)

                valores_entrada[columna] = valor

    st.markdown("---")
    enviar = st.form_submit_button(
        "✨ Analizar estudiante",
        type="primary",
        use_container_width=True
    )


# ============================================================
# 7. RESULTADOS
# ============================================================

if enviar:
    try:
        datos_estudiante = pd.DataFrame(
            [valores_entrada],
            columns=columnas_modelo
        )

        datos_transformados = preprocesador.transform(datos_estudiante)
        probabilidades = modelo.predict(datos_transformados, verbose=0)[0]

        indice_prediccion = int(np.argmax(probabilidades))
        resultado = clases[indice_prediccion]
        resultado_traducido = CLASS_LABELS.get(resultado, resultado)
        confianza = float(probabilidades[indice_prediccion]) * 100

        st.markdown("---")
        st.markdown('<div class="section-heading">📊 Resultado del análisis</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-subtitle">Estimación generada por el modelo de aprendizaje automático.</div>',
            unsafe_allow_html=True
        )

        if resultado == "Dropout":
            st.error("El modelo identifica un posible riesgo de deserción estudiantil.")
        elif resultado == "Enrolled":
            st.info("El modelo estima que el estudiante continúa matriculado.")
        else:
            st.success("El modelo estima un resultado asociado a la graduación.")

        st.markdown(f"""
        <div class="result-card">
            <div class="result-title">Resultado estimado</div>
            <div class="result-value">{resultado_traducido}</div>
            <div style="color:#aab8cc; margin-top:.35rem;">
                Probabilidad de la clase estimada: <b>{confianza:.2f}%</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        resultados = pd.DataFrame({
            "Resultado": [CLASS_LABELS.get(clase, clase) for clase in clases],
            "Probabilidad (%)": [
                round(float(p) * 100, 2) for p in probabilidades
            ]
        })

        st.markdown("#### Probabilidad por resultado")
        st.dataframe(
            resultados,
            hide_index=True,
            use_container_width=True
        )

        st.markdown("#### Distribución de probabilidades")

        # Gráfico horizontal para evitar que los nombres de las clases
        # queden girados, truncados o cortados en pantallas pequeñas.
        etiquetas_grafico = resultados["Resultado"].tolist()
        valores_grafico = resultados["Probabilidad (%)"].tolist()
        paleta_cun = ["#22A6C7", "#8B3FA0", "#55208C"]

        fig, ax = plt.subplots(figsize=(10, 3.8))
        fig.patch.set_facecolor("#111B42")
        ax.set_facecolor("#111B42")

        barras = ax.barh(
            etiquetas_grafico,
            valores_grafico,
            color=paleta_cun[:len(valores_grafico)],
            height=0.55
        )

        ax.invert_yaxis()
        ax.set_xlim(0, 108)
        ax.set_xlabel("Probabilidad (%)", color="#E9E7F5", labelpad=10)
        ax.tick_params(axis="x", colors="#D8D8EA", labelsize=9)
        ax.tick_params(axis="y", colors="#FFFFFF", labelsize=10, pad=10)
        ax.grid(axis="x", linestyle="--", alpha=0.22, color="#C7C4E8")
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)

        for barra, valor in zip(barras, valores_grafico):
            ax.text(
                min(valor + 1.2, 102),
                barra.get_y() + barra.get_height() / 2,
                f"{valor:.2f}%",
                va="center",
                ha="left",
                color="#FFFFFF",
                fontsize=10,
                fontweight="bold"
            )

        ax.set_title(
            "Probabilidad estimada por categoría",
            color="#FFFFFF",
            fontsize=13,
            fontweight="bold",
            loc="left",
            pad=16
        )

        fig.tight_layout(pad=2.0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.caption(
            "Las probabilidades son estimaciones del modelo y no representan "
            "una certeza sobre el futuro académico del estudiante."
        )

    except Exception as error:
        st.error("No fue posible generar la predicción.")
        st.exception(error)


# ============================================================
# 8. INFORMACIÓN TÉCNICA Y PIE DE PÁGINA
# ============================================================

with st.expander("ℹ️ Consultar información técnica del modelo"):
    st.write("**Algoritmo:** Red neuronal artificial MLP")
    st.write(f"**Variables de entrada:** {len(columnas_modelo)}")
    st.write(f"**Variables categóricas:** {len(variables_categoricas)}")
    st.write(f"**Variables numéricas:** {len(variables_numericas)}")
    st.write(
        "**Clases de salida:** "
        + ", ".join(CLASS_LABELS.get(c, c) for c in clases)
    )
    st.write(f"**Dimensiones de entrada:** {modelo.input_shape[1]}")
    st.write(f"**Dimensiones de salida:** {modelo.output_shape[1]}")

st.markdown(
    '<div class="footnote">Proyecto académico de Aprendizaje Automático · '
    'Predicción de deserción y éxito académico estudiantil</div>',
    unsafe_allow_html=True
)
