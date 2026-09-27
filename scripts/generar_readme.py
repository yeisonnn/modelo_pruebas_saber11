"""Genera README.md con los resultados reales (lee models/metadata.json y reports/tablas/resultados_test.csv)."""
import json
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
FENCE = "`" * 3   # Delimitador de bloque de código en Markdown (se arma así para no romper este archivo)

meta = json.loads((RAIZ / "models" / "metadata.json").read_text(encoding="utf-8"))
test = pd.read_csv(RAIZ / "reports" / "tablas" / "resultados_test.csv")
m = meta["metricas_test"]
filas = "\n".join(f"| {r.modelo} | {r.accuracy:.4f} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} | {r.roc_auc:.4f} |"
                  for r in test.itertuples())
python_despliegue = meta["versiones"]["python"].rsplit(".", 1)[0]   # "3.12.10" -> "3.12"

README = f"""# 🎓 Predicción de alto desempeño en Saber 11 — Antioquia 2022-2

> 📚 **Proyecto netamente educativo** de nivel maestría: es el proyecto práctico de la asignatura y material de estudio
> de CRISP-DM y aprendizaje supervisado. No es una herramienta oficial del ICFES ni debe usarse para decisiones sobre
> estudiantes reales.

Proyecto integrador de la Maestría en Ciencia de Datos con metodología **CRISP-DM**. Se predice si un estudiante obtendrá
un **puntaje global ≥ 300** en la prueba Saber 11 a partir de información socioeconómica y del colegio disponible antes del
examen.

| | |
|---|---|
| **Fuente** | ICFES — *Resultados únicos Saber 11*, Datos Abiertos Colombia ([kgxf-xxbe]({meta['fuente']})) |
| **Alcance** | Periodo 2022-2, colegios de Antioquia |
| **Problema** | Clasificación binaria · métrica principal: F1 de la clase positiva |
| **Modelo final** | {meta['modelo_final']} · umbral de decisión t = {meta['umbral_decision']:.3f} |
| **Capa prescriptiva** | Bandas de acompañamiento: prioridad de nivelación (p < t/2), refuerzo focalizado (t/2 ≤ p < t) y alto potencial (p ≥ t) |
| **Test** | F1 = {m['f1']:.3f} · Recall = {m['recall']:.3f} · Precision = {m['precision']:.3f} · ROC-AUC = {m['roc_auc']:.3f} |
| **App** | [COMPLETAR: URL pública de Streamlit.io] |
| **Colab** | [COMPLETAR: enlace público del notebook 02 (entregable)] |

## Resultados en el conjunto de prueba (30 %)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
{filas}

## Notebooks

| Notebook | Uso |
|---|---|
| `notebooks/02_Entregable_Proyecto_Integrador_Saber11.ipynb` | **Entregable final** (enlace de Colab): las 6 fases CRISP-DM con explicación, tablas, gráficos, métricas, modelo prescriptivo y conclusiones |
| `notebooks/01_Material_de_estudio_Saber11.ipynb` | Material de estudio: la misma secuencia con teoría ampliada (📘), interpretaciones y preguntas de repaso (🧠) |

En ambos, cada celda de código tiene comentarios que explican el *qué* y el *porqué*, y debajo una 🔎 **Interpretación**
con las cifras obtenidas; en el entregable, además, cada celda va precedida de **¿Qué hacemos y por qué?**

## Estructura

{FENCE}
app.py                      App Streamlit
requirements.txt            Dependencias de la app (versiones fijadas)
models/                     Pipeline serializado (joblib), metadata e importancias
notebooks/                  Entregable (02), material de estudio (01) y sus fuentes py:percent
data/raw/                   Dataset original comprimido (API datos.gov.co)
data/nuevos/                Perfiles nuevos y 1.000 registros no vistos para probar la app
reports/                    Reporte ydata-profiling (HTML), figuras y tablas
scripts/                    Construcción del notebook, README e informe
tests/                      Pruebas automáticas (notebook, modelo y app)
{FENCE}

## Ejecutar en local

{FENCE}bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-notebook.txt
.venv/Scripts/streamlit run app.py
{FENCE}

En Linux o macOS la ruta del intérprete es `.venv/bin/python` en lugar de `.venv/Scripts/python`.

Para reconstruir y ejecutar un notebook desde sus fuentes (unos 30 minutos cada uno):

{FENCE}bash
.venv/Scripts/python scripts/construir_notebook.py --notebook entregable --ejecutar   # 02
.venv/Scripts/python scripts/construir_notebook.py --notebook estudio --ejecutar      # 01
{FENCE}

## Desplegar en Streamlit Community Cloud

1. Subir este repositorio a GitHub (público).
2. Entrar a https://share.streamlit.io → **Create app** → elegir el repositorio, rama `main` y archivo `app.py`.
3. En *Advanced settings* elegir **Python {python_despliegue}** y desplegar.

## Uso ético

El modelo refleja asociaciones socioeconómicas estructurales, no causas. Aun con fines académicos, sus resultados no deben
usarse para excluir estudiantes ni para decisiones individuales adversas.

## Reproducibilidad

Semilla 42 · versiones: {', '.join(f'{k} {v}' for k, v in meta['versiones'].items())} · entrenado el {meta['fecha_entrenamiento']}.
"""
(RAIZ / "README.md").write_text(README, encoding="utf-8")
print("README.md generado")
