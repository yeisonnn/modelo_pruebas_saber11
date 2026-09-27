# 🎓 Predicción de alto desempeño en Saber 11 — Antioquia 2022-2

> 📚 **Proyecto netamente educativo** de nivel maestría: es el proyecto práctico de la asignatura y material de estudio
> de CRISP-DM y aprendizaje supervisado. No es una herramienta oficial del ICFES ni debe usarse para decisiones sobre
> estudiantes reales.

Proyecto integrador de la Maestría en Ciencia de Datos con metodología **CRISP-DM**. Se predice si un estudiante obtendrá
un **puntaje global ≥ 300** en la prueba Saber 11 a partir de información socioeconómica y del colegio disponible antes del
examen.

| | |
|---|---|
| **Fuente** | ICFES — *Resultados únicos Saber 11*, Datos Abiertos Colombia ([kgxf-xxbe](https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe)) |
| **Alcance** | Periodo 2022-2, colegios de Antioquia |
| **Problema** | Clasificación binaria · métrica principal: F1 de la clase positiva |
| **Modelo final** | Regresión Logística (ajustado + umbral) · umbral de decisión t = 0.615 |
| **Capa prescriptiva** | Bandas de acompañamiento: prioridad de nivelación (p < t/2), refuerzo focalizado (t/2 ≤ p < t) y alto potencial (p ≥ t) |
| **Test** | F1 = 0.473 · Recall = 0.579 · Precision = 0.400 · ROC-AUC = 0.798 |
| **App** | [COMPLETAR: URL pública de Streamlit.io] |
| **Colab** | [COMPLETAR: enlace público del notebook 02 (entregable)] |

## Resultados en el conjunto de prueba (30 %)

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Regresión Logística (ajustado + umbral) | 0.7789 | 0.3996 | 0.5795 | 0.4730 | 0.7975 |
| Random Forest (ajustado + umbral) | 0.7723 | 0.3916 | 0.5957 | 0.4726 | 0.7967 |
| SVM (RBF) (ajustado + umbral) | 0.7663 | 0.3845 | 0.6067 | 0.4707 | 0.7799 |
| Random Forest (ajustado) | 0.7097 | 0.3386 | 0.7294 | 0.4625 | 0.7967 |
| XGBoost (base) | 0.7036 | 0.3353 | 0.7443 | 0.4623 | 0.7995 |
| Regresión Logística (ajustado) | 0.7041 | 0.3345 | 0.7359 | 0.4599 | 0.7975 |
| Random Forest (base) | 0.7079 | 0.3362 | 0.7249 | 0.4594 | 0.7967 |
| Regresión Logística (base) | 0.7026 | 0.3334 | 0.7378 | 0.4593 | 0.7975 |
| SVM (RBF) (base) | 0.6961 | 0.3292 | 0.7469 | 0.4570 | 0.7904 |
| SVM (RBF) (ajustado) | 0.6988 | 0.3298 | 0.7359 | 0.4555 | 0.7799 |
| Votación (soft) (base) | 0.6990 | 0.3288 | 0.7281 | 0.4531 | 0.7931 |
| Bagging (árboles) (base) | 0.6962 | 0.3170 | 0.6703 | 0.4304 | 0.7616 |
| Árbol de Decisión (base) | 0.6931 | 0.3140 | 0.6690 | 0.4274 | 0.7480 |
| KNN (base) | 0.6472 | 0.2914 | 0.7404 | 0.4182 | 0.7601 |

## Notebooks

| Notebook | Uso |
|---|---|
| `notebooks/02_Entregable_Proyecto_Integrador_Saber11.ipynb` | **Entregable final** (enlace de Colab): las 6 fases CRISP-DM con explicación, tablas, gráficos, métricas, modelo prescriptivo y conclusiones |
| `notebooks/01_Material_de_estudio_Saber11.ipynb` | Material de estudio: la misma secuencia con teoría ampliada (📘), interpretaciones y preguntas de repaso (🧠) |

En ambos, cada celda de código tiene comentarios que explican el *qué* y el *porqué*, y debajo una 🔎 **Interpretación**
con las cifras obtenidas; en el entregable, además, cada celda va precedida de **¿Qué hacemos y por qué?**

## Estructura

```
app.py                      App Streamlit
requirements.txt            Dependencias de la app (versiones fijadas)
models/                     Pipeline serializado (joblib), metadata e importancias
notebooks/                  Entregable (02), material de estudio (01) y sus fuentes py:percent
data/raw/                   Dataset original comprimido (API datos.gov.co)
data/nuevos/                Perfiles nuevos y 1.000 registros no vistos para probar la app
reports/                    Reporte ydata-profiling (HTML), figuras y tablas
scripts/                    Construcción del notebook, README e informe
tests/                      Pruebas automáticas (notebook, modelo y app)
```

## Ejecutar en local

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-notebook.txt
.venv/Scripts/streamlit run app.py
```

En Linux o macOS la ruta del intérprete es `.venv/bin/python` en lugar de `.venv/Scripts/python`.

Para reconstruir y ejecutar un notebook desde sus fuentes (unos 30 minutos cada uno):

```bash
.venv/Scripts/python scripts/construir_notebook.py --notebook entregable --ejecutar   # 02
.venv/Scripts/python scripts/construir_notebook.py --notebook estudio --ejecutar      # 01
```

## Desplegar en Streamlit Community Cloud

1. Subir este repositorio a GitHub (público).
2. Entrar a https://share.streamlit.io → **Create app** → elegir el repositorio, rama `main` y archivo `app.py`.
3. En *Advanced settings* elegir **Python 3.12** y desplegar.

## Uso ético

El modelo refleja asociaciones socioeconómicas estructurales, no causas. Aun con fines académicos, sus resultados no deben
usarse para excluir estudiantes ni para decisiones individuales adversas.

## Reproducibilidad

Semilla 42 · versiones: python 3.12.10, scikit-learn 1.9.1, imbalanced-learn 0.14.2, xgboost 3.4.1, pandas 2.3.3, numpy 2.3.5 · entrenado el 2026-09-26.
