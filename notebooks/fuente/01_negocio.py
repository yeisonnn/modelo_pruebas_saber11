# %% [markdown]
# ## 1. Entendimiento del negocio
#
# ### 1.1 Contexto
# La prueba **Saber 11** es el examen de Estado que presentan quienes terminan la educación media en Colombia. La aplica el
# ICFES y evalúa cinco áreas (Lectura crítica, Matemáticas, Sociales y ciudadanas, Ciencias naturales e Inglés).
# El **puntaje global (0–500)** se usa en la admisión a la educación superior, en la asignación de becas y créditos y en la
# clasificación de los colegios. Junto con los resultados, el ICFES recoge información socioeconómica del hogar y
# características del colegio, lo que permite estudiar las brechas de desempeño.
#
# En Antioquia, el conjunto publicado para 2022-2 trae 144.766 registros, pero cerca de la mitad son duplicados exactos
# (se depuran en 3.3): quedan unos 72 mil estudiantes. Como **caso de estudio académico**
# se plantea el escenario de una secretaría de educación que quisiera identificar temprano a los estudiantes con baja
# probabilidad de alto desempeño para focalizar nivelación, preparación y orientación vocacional.
#
# ### 1.2 Problema predictivo
# ¿Es posible estimar, **con información disponible antes del examen** (hogar, estudiante y colegio), la probabilidad de que
# un estudiante de Antioquia obtenga un **puntaje global ≥ 300**?
# - **Variable objetivo:** `alto_desempeno` = 1 si `punt_global ≥ 300`; 0 en otro caso.
# - **Unidad de análisis:** estudiante evaluado en 2022-2 en un colegio de Antioquia.
# - **Umbral:** 300 puntos equivale al 60 % de la escala y separa aproximadamente al 17 % superior (se verifica en la sección 2.3).
#
# ### 1.3 Objetivos de minería de datos
# 1. Construir un clasificador supervisado con un **F1 de la clase positiva** superior a la línea base (regresión logística).
# 2. Comparar 5 modelos clásicos y 3 de ensamble con validación cruzada estratificada sobre el 70 % de entrenamiento.
# 3. Identificar las variables más asociadas al alto desempeño (importancia por permutación).
# 4. Desplegar el pipeline en una aplicación web (Streamlit) como demostración académica.
#
# ### 1.4 Diseño de la solución
# | Tipo de análisis | Tipo de aprendizaje | Modelos propuestos | Métrica principal |
# |---|---|---|---|
# | Predictivo | Supervisado (clasificación binaria) | Regresión Logística, KNN, Árbol de Decisión, SVM (RBF), Random Forest; ensambles: Votación suave, Bagging, XGBoost | **F1** de la clase positiva (complementarias: Recall, Precision, ROC-AUC, PR-AUC) |
#
# **¿Por qué F1?** Con cerca de 17 % de positivos, un modelo que siempre predice "no alto" tendría 83 % de *accuracy* sin
# ningún valor práctico. F1 equilibra precisión y exhaustividad sobre la clase de interés (se profundiza en la fase 5).
#
# **Uso ético:** el modelo describe asociaciones estructurales (socioeconómicas), no causas. Aun en un escenario real serviría
# para asignar apoyos, **nunca** para excluir estudiantes o tomar decisiones individuales adversas.

# %% [markdown]
# ### 🧠 Para profundizar — Fase 1: Entendimiento del negocio
# **Ideas clave**
# - Un proyecto de analítica empieza por una **pregunta de negocio** y un **criterio de éxito medible**, no por un algoritmo.
# - Traducir el problema a un objetivo de minería implica decidir la unidad de análisis, la variable objetivo y la métrica.
# - Solo pueden usarse variables **disponibles en el momento de la predicción**; lo demás es fuga de información.
#
# **Preguntas de repaso**
# 1. ¿Por qué *accuracy* es engañosa cuando la clase positiva es el 17 % de los datos?
# 2. ¿Qué cambiaría en el diseño si el problema se planteara como regresión del puntaje global?
# 3. ¿Qué riesgos éticos aparecen al predecir desempeño académico con variables socioeconómicas?
#
# **Ejercicios**
# - Reformula el problema con otro umbral (250 o 350) y anticipa cómo cambiarían el desbalance y la métrica adecuada.
# - Redacta un criterio de éxito de negocio (no técnico) para este modelo.
#
# **Referencias**
# - Chapman, P. et al. (2000). *CRISP-DM 1.0: Step-by-step data mining guide*. SPSS.
# - Wirth, R. y Hipp, J. (2000). CRISP-DM: Towards a standard process model for data mining. *4th Int. Conf. on the Practical Applications of Knowledge Discovery and Data Mining*.
# - Provost, F. y Fawcett, T. (2013). *Data Science for Business*. O'Reilly (cap. 2: problemas de negocio y minería de datos).
