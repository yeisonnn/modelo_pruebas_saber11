"""App Streamlit: predicción de alto desempeño (≥ 300) en Saber 11 — Antioquia 2022-2."""
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
RUTA_MODELO = BASE / "models" / "pipeline_saber11.joblib"
RUTA_METADATA = BASE / "models" / "metadata.json"
RUTA_IMPORTANCIAS = BASE / "models" / "importancias.csv"
RUTA_EJEMPLO = BASE / "data" / "nuevos" / "registros_no_vistos_1000.csv"
RUTA_MATRIZ = BASE / "reports" / "figures" / "matriz_confusion_final.png"
RUTA_DATOS_LIMPIOS = BASE / "data" / "processed" / "saber11_antioquia_limpio.csv.gz"
# Configuración del modelo final según el notebook (secciones 0.3, 3.5 y 4.3). Solo se usa si hay que reconstruirlo
SEMILLA, N_MUESTRA, PROPORCION_PRUEBA = 42, 30_000, 0.30
C_FINAL = 0.14445251022763064          # Mejor C de la búsqueda aleatoria de la regresión logística (sección 4.3.2)
OBJETIVO = "alto_desempeno"
NO_ORDINALES = ("No sabe", "No Aplica")   # Respuestas sin lugar en la escala ordinal: el pipeline las trata como faltantes
GRUPOS = {"Estudiante": ["edad", "estu_genero"],
          "Familia": ["fami_estratovivienda", "fami_educacionmadre", "fami_educacionpadre", "fami_personashogar",
                      "fami_cuartoshogar", "fami_tieneinternet", "fami_tienecomputador", "fami_tieneautomovil",
                      "fami_tienelavadora"],
          "Colegio": ["cole_naturaleza", "cole_jornada", "cole_area_ubicacion", "cole_bilingue", "cole_caracter",
                      "cole_genero", "cole_sede_principal", "cole_valle_aburra"]}

st.set_page_config(page_title="Saber 11 · Alto desempeño", page_icon="🎓", layout="wide")


def reconstruir_modelo(meta):
    """Reentrena el pipeline final con los mismos datos, partición, preprocesamiento e hiperparámetros del notebook."""
    datos = pd.read_csv(RUTA_DATOS_LIMPIOS)
    # Misma muestra de modelado (30.000) y misma partición 70/30 estratificada que el notebook (sección 3.5.1)
    muestra, _ = train_test_split(datos, train_size=N_MUESTRA, stratify=datos[OBJETIVO], random_state=SEMILLA)
    X_train, _, y_train, _ = train_test_split(muestra[meta["features"]], muestra[OBJETIVO], test_size=PROPORCION_PRUEBA,
                                              stratify=muestra[OBJETIVO], random_state=SEMILLA)
    # Orden de cada variable ordinal: el de la metadata sin las respuestas que no son un nivel (sección 3.5.2)
    orden = [[c for c in meta["categorias"][v] if c not in NO_ORDINALES] for v in meta["variables_ordinales"]]
    preprocesador = ColumnTransformer([
        ("num", Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())]),
         meta["variables_numericas"]),
        ("ord", Pipeline([("codificar", OrdinalEncoder(categories=orden, handle_unknown="use_encoded_value",
                                                       unknown_value=np.nan)),
                          ("imputar", SimpleImputer(strategy="median", add_indicator=True)),
                          ("escalar", StandardScaler())]), meta["variables_ordinales"]),
        ("nom", Pipeline([("imputar", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),
                          ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
         meta["variables_nominales"])], remainder="drop")
    # Preprocesamiento -> submuestreo -> regresión logística ajustada (el umbral de decisión viene de la metadata)
    pipeline = ImbPipeline([("preprocesamiento", preprocesador),
                            ("balanceo", RandomUnderSampler(random_state=SEMILLA)),
                            ("modelo", LogisticRegression(C=C_FINAL, max_iter=2000, random_state=SEMILLA))])
    return pipeline.fit(X_train, y_train)


@st.cache_resource
def cargar_modelo(_meta):
    """Carga el pipeline serializado; si las librerías instaladas no pueden leerlo, lo reconstruye (tarda unos segundos)."""
    try:
        # joblib usa pickle: solo se carga el artefacto propio de este repositorio (fuente confiable), nunca archivos del usuario
        return joblib.load(RUTA_MODELO), None
    except Exception as error:   # Un pickle falla con distintos errores según la versión de las librerías
        return reconstruir_modelo(_meta), f"{type(error).__name__}: {str(error)[:200]}"


@st.cache_data
def cargar_metadata():
    return json.loads(RUTA_METADATA.read_text(encoding="utf-8"))


def puntaje(modelo, X):
    return modelo.predict_proba(X)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X)


def predecir(modelo, meta, df):
    s = puntaje(modelo, df[meta["features"]])
    return s, (s >= meta["umbral_decision"]).astype(int)


def banda(s, umbral):
    """Regla prescriptiva del notebook (sección 1.3): banda de acompañamiento según el puntaje y el umbral t."""
    if s >= umbral:
        return "🟢 Alto potencial"
    return "🟠 Refuerzo focalizado" if s >= umbral / 2 else "🔴 Prioridad de nivelación"


def interpretar(s, umbral):
    if s >= umbral:
        return "El perfil se parece al de estudiantes que alcanzan alto desempeño (≥ 300 puntos)."
    if s >= umbral / 2:
        return "Probabilidad intermedia: con refuerzo focalizado podría alcanzar el umbral."
    return "Baja probabilidad de alto desempeño: perfil prioritario para programas de nivelación y acompañamiento."


meta = cargar_metadata()
modelo, error_carga = cargar_modelo(meta)
etiquetas, categorias = meta["etiquetas"], meta["categorias"]
es_probabilidad = meta["tipo_puntaje"] == "probabilidad"

st.title("🎓 Predicción de alto desempeño en Saber 11")
st.caption(f"Antioquia · periodo 2022-2 · Modelo: {meta['modelo_final']} · Fuente: ICFES — Datos Abiertos Colombia")
st.info("📚 Proyecto académico de maestría con fines exclusivamente educativos (metodología CRISP-DM). "
        "No es una herramienta oficial del ICFES ni debe usarse para tomar decisiones sobre estudiantes reales.")
if error_carga:
    # Transparencia: el modelo en uso se reentrenó en este servidor con la configuración del notebook
    st.warning("El archivo del modelo no es compatible con las versiones de librerías de este servidor, así que se "
               "reconstruyó con los mismos datos, partición, preprocesamiento e hiperparámetros del notebook. "
               f"Motivo técnico: {error_carga}")

tab_individual, tab_lote, tab_modelo = st.tabs(["Predicción individual", "Predicción por lotes", "Sobre el modelo"])

with tab_individual:
    with st.form("formulario"):
        valores = {}
        for columna, (grupo, variables) in zip(st.columns(3), GRUPOS.items()):
            with columna:
                st.subheader(grupo)
                for var in variables:
                    if var == "edad":
                        valores[var] = st.number_input(etiquetas[var], min_value=meta["rango_edad"][0],
                                                       max_value=meta["rango_edad"][1],
                                                       value=int(round(meta.get("edad_mediana", 17))), step=1)
                    else:
                        # El formulario arranca con el valor más frecuente (perfil típico) si la metadata lo trae
                        opciones = categorias[var]
                        defecto = meta.get("valores_por_defecto", {}).get(var)
                        indice = opciones.index(defecto) if defecto in opciones else 0
                        valores[var] = st.selectbox(etiquetas[var], opciones, index=indice)
        enviado = st.form_submit_button("Predecir", type="primary")
    if enviado:
        s, clase = predecir(modelo, meta, pd.DataFrame([valores]))
        etiqueta = "Probabilidad de alto desempeño" if es_probabilidad else "Puntaje de decisión"
        st.metric(etiqueta, f"{s[0]:.1%}" if es_probabilidad else f"{s[0]:.3f}",
                  help=f"Umbral de decisión: {meta['umbral_decision']:.3f}")
        if es_probabilidad:
            st.progress(float(np.clip(s[0], 0, 1)))
        (st.success if clase[0] == 1 else st.info)(
            f"Predicción: **{'Alto desempeño' if clase[0] == 1 else 'No alto desempeño'}** — "
            f"{interpretar(s[0], meta['umbral_decision'])}")
        st.markdown(f"**Banda de acompañamiento sugerida:** {banda(s[0], meta['umbral_decision'])}")

with tab_lote:
    st.markdown("Suba un CSV con las columnas de entrada del modelo o use los datos de ejemplo.")
    st.code(", ".join(meta["features"]), language=None)
    archivo = st.file_uploader("Archivo CSV", type=["csv"])
    if st.button("Usar 1.000 registros de ejemplo no vistos por el modelo", key="btn_ejemplo"):
        st.session_state["usar_ejemplo"] = True
    df_lote = pd.read_csv(archivo) if archivo is not None else (
        pd.read_csv(RUTA_EJEMPLO) if st.session_state.get("usar_ejemplo") else None)
    if df_lote is not None:
        faltantes = [c for c in meta["features"] if c not in df_lote.columns]
        if faltantes:
            st.error("Faltan columnas: " + ", ".join(faltantes))
        else:
            s, clase = predecir(modelo, meta, df_lote)
            resultado = df_lote.assign(puntaje=np.round(s, 4),
                                       prediccion=np.where(clase == 1, "Alto desempeño", "No alto"),
                                       banda=[banda(v, meta["umbral_decision"]) for v in s])
            st.session_state["n_lote"] = len(resultado)
            c1, c2, c3 = st.columns(3)
            c1.metric("Registros evaluados", f"{len(resultado):,}")
            c2.metric("Predichos con alto desempeño", f"{clase.mean():.1%}")
            if "alto_desempeno" in df_lote.columns:
                c3.metric("F1 frente al valor real", f"{f1_score(df_lote['alto_desempeno'], clase):.3f}")
            # Resumen prescriptivo: cuántos estudiantes quedan en cada banda de acompañamiento
            st.markdown("**Bandas de acompañamiento (% de registros):** " + " · ".join(
                f"{b} {p:.1%}" for b, p in resultado["banda"].value_counts(normalize=True).items()))
            st.dataframe(resultado, height=380)
            st.download_button("Descargar predicciones (CSV)", resultado.to_csv(index=False).encode("utf-8"),
                               file_name="predicciones_saber11.csv", mime="text/csv")

with tab_modelo:
    st.subheader("Desempeño en el conjunto de prueba (30 %)")
    m = meta["metricas_test"]
    for columna, (nombre, clave) in zip(st.columns(6), [("Accuracy", "accuracy"), ("Precision", "precision"),
                                                        ("Recall", "recall"), ("F1", "f1"), ("ROC-AUC", "roc_auc"),
                                                        ("PR-AUC", "pr_auc")]):
        columna.metric(nombre, f"{m[clave]:.3f}")
    izquierda, derecha = st.columns(2)
    if RUTA_MATRIZ.exists():
        izquierda.image(str(RUTA_MATRIZ), caption="Matriz de confusión (test)", width=460)
    if RUTA_IMPORTANCIAS.exists():
        importancias = pd.read_csv(RUTA_IMPORTANCIAS)
        importancias["variable"] = importancias["variable"].map(etiquetas).fillna(importancias["variable"])
        derecha.markdown("**Importancia por permutación (caída del F1)**")
        # Barras horizontales ordenadas por importancia (bar_chart las ordenaría alfabéticamente)
        grafico = alt.Chart(importancias).mark_bar(color="#1F4E79").encode(
            x=alt.X("importancia:Q", title="Disminución del F1"),
            y=alt.Y("variable:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),   # Etiquetas completas
            tooltip=["variable", alt.Tooltip("importancia:Q", format=".4f")])
        derecha.altair_chart(grafico)
    st.warning("Uso ético: el modelo refleja asociaciones socioeconómicas, no causas. Aun con fines académicos, "
               "sus resultados no deben usarse para excluir estudiantes ni para decisiones individuales adversas.")
    st.caption(meta["proposito"])
    st.caption(f"Objetivo: {meta['objetivo']} · Entrenamiento: {meta['n_entrenamiento']:,} registros · "
               f"Versiones: {meta['versiones']}")
