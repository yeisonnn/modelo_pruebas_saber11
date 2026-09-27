# %% [markdown]
# # Proyecto Integrador — Metodología CRISP-DM
# ## Predicción del alto desempeño en la prueba Saber 11 · Antioquia 2022-2
#
# | Ficha técnica | |
# |---|---|
# | **Programa** | Maestría en Ciencia de Datos — proyecto integrador |
# | **Integrantes** | [COMPLETAR] |
# | **Tipo de problema** | Clasificación binaria supervisada (+ capa prescriptiva) |
# | **Fuente de datos** | ICFES — *Resultados únicos Saber 11*, Datos Abiertos Colombia (`kgxf-xxbe`) |
# | **Enlace al dataset** | https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe |
# | **Modelo final** | Regresión logística con umbral de decisión ajustado (0,615): F1 = 0,473 en prueba (sección 5.2) |
# | **Capa prescriptiva** | Bandas de acompañamiento: prioridad de nivelación, refuerzo focalizado y alto potencial (sección 6.2) |
# | **Aplicación web** | [COMPLETAR: URL pública de Streamlit.io] |
#
# > ⚠️ **Aviso académico:** proyecto con fines exclusivamente educativos. No es una herramienta oficial del ICFES y sus
# > resultados no deben usarse para tomar decisiones sobre estudiantes reales.
#
# **Contenido:** 0. Configuración · 1. Entendimiento del negocio · 2. Entendimiento de los datos · 3. Preparación de los datos ·
# 4. Modelamiento · 5. Evaluación · 6. Despliegue · 7. Conclusiones y recomendaciones
#
# **Cómo leer este cuaderno:** cada paso tiene una celda **"¿Qué hacemos y por qué?"**, el código comentado y una celda
# **🔎 Interpretación** con la lectura de los resultados. La teoría ampliada (fórmulas, referencias y ejercicios) está en el
# cuaderno complementario `01_Material_de_estudio_Saber11.ipynb`.
#
# **Ejecución:** el cuaderno se comparte **ya ejecutado**, así que todas las tablas y gráficos se pueden leer sin volver a
# correrlo. Una ejecución completa tarda unos 30 minutos en un equipo de 4 núcleos y puede superar una hora en Colab; la
# etapa más larga es la búsqueda de hiperparámetros del SVM (sección 4.3).

# %% [markdown]
# ## 0. Configuración del entorno

# %% [markdown]
# **¿Qué hacemos y por qué?** Instalamos, solo si el cuaderno corre en Google Colab, las versiones exactas de las librerías con
# las que se entrenó el modelo. Así cualquier persona obtiene los mismos resultados y el modelo serializado carga sin errores.
# Si la instalación reemplaza una librería que Colab ya tenía cargada, la celda pide reiniciar la sesión y volver a
# ejecutar desde aquí.

# %%
#@copiar 0.1

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra que, en local, el cuaderno usa el entorno virtual `.venv` y no instala nada. En Colab instalaría las versiones exactas de scikit-learn, imbalanced-learn, xgboost y ydata-profiling, más `setuptools<81`, que ydata-profiling necesita para importar `pkg_resources`, y avisaría si hay que reiniciar la sesión porque pip reemplazó una librería ya cargada.
# En general, fijar versiones es la primera condición de la reproducibilidad: el pipeline se serializa con una versión concreta de scikit-learn (fase 6), y cargarlo con otra versión podría fallar o cambiar las predicciones.

# %% [markdown]
# **¿Qué hacemos y por qué?** Importamos las librerías agrupadas por propósito: manipulación de datos, visualización,
# estadística, preprocesamiento, modelos, métricas y balanceo. Imprimir las versiones deja constancia del entorno.

# %%
#@copiar 0.2

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra que todas las librerías cargan sin errores y deja registradas sus versiones: Python 3.12.10, pandas 2.3.3, numpy 2.3.5, scikit-learn 1.9.1, imbalanced-learn 0.14.2 y xgboost 3.4.1.
# En general, registrar las versiones en el propio cuaderno permite repetir el análisis más adelante. Se usa pandas 2.x, y no 3.x, porque ydata-profiling 4.18 exige `pandas<3`: una librería del proyecto restringe la versión de otra.

# %% [markdown]
# **¿Qué hacemos y por qué?** Cargamos tres herramientas que usa solo este cuaderno: la curva de calibración, la curva
# Precision–Recall por umbral y las cajas del diagrama de flujo de la sección 6.

# %%
# 0.2.1 Librerías adicionales del entregable: diagnóstico del clasificador y diagramas
from sklearn.calibration import calibration_curve        # Curva de calibración de probabilidades
from sklearn.metrics import precision_recall_curve        # Precision y Recall para todos los umbrales posibles
from matplotlib.patches import FancyBboxPatch             # Cajas redondeadas del diagrama de flujo (6.3)
from IPython.display import display                       # Mostrar varias tablas en una misma celda
print("Librerías adicionales cargadas: calibration_curve, precision_recall_curve, FancyBboxPatch, display")

# %% [markdown]
# 🔎 **Interpretación:** La salida confirma que se cargaron las cuatro herramientas adicionales del entregable: `calibration_curve` y `precision_recall_curve` para el diagnóstico del clasificador (5.3), `FancyBboxPatch` para el diagrama de flujo (6.3) y `display` para mostrar varias tablas en una misma celda.
# En general, separar estas importaciones de las del bloque 0.2 deja claro qué se añadió al cuaderno de estudio para construir el entregable, sin cambiar nada del flujo original.

# %% [markdown]
# **¿Qué hacemos y por qué?** Fijamos en un solo lugar los parámetros del proyecto: la semilla, el umbral de negocio (300
# puntos), el tamaño de la muestra de modelado y la partición 70/30. También creamos las carpetas de salida.

# %%
#@copiar 0.3

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra los parámetros globales del proyecto: semilla 42, umbral de negocio de 300 puntos, muestra de modelado de 30.000 registros, partición 70/30 y un único objeto `CV` con 5 folds estratificados.
# En general, que todos los modelos compartan exactamente los mismos folds es lo que permite comparar sus F1 fold a fold con pruebas **pareadas** en la fase 5. Las carpetas de salida se crean con rutas relativas, así que el cuaderno funciona igual en local y en Colab.
