# %% [markdown]
# ## 1. Entendimiento del negocio
#
# ### 1.1 Contexto: la prueba Saber 11 en Antioquia
# El **Instituto Colombiano para la Evaluación de la Educación (ICFES)** aplica cada año la prueba **Saber 11**, el examen de
# Estado de la educación media. Evalúa cinco áreas (Lectura crítica, Matemáticas, Sociales y ciudadanas, Ciencias naturales e
# Inglés) y entrega un **puntaje global de 0 a 500**. Además, recoge información socioeconómica del hogar y características
# del colegio de cada estudiante.
#
# - **Cobertura del caso:** colegios de Antioquia, periodo 2022-2 (calendario A).
# - **Tamaño:** 144.766 registros publicados; cerca de 72 mil estudiantes únicos tras depurar duplicados (sección 3.3).
# - **Territorio:** 125 municipios, del Valle de Aburrá a las subregiones rurales del departamento.
# - **Usos del puntaje:** admisión a la educación superior, becas y créditos educativos y clasificación de colegios.
#
# #### 1.1.1 Problemática
# Los resultados de Saber 11 muestran **brechas persistentes** entre estudiantes de distintos contextos: estrato, educación
# de los padres, zona urbana o rural y naturaleza del colegio. En el escenario del caso de estudio, una secretaría de educación
# quiere **identificar temprano** qué estudiantes tienen baja probabilidad de alcanzar un alto desempeño, para focalizar
# programas de nivelación, refuerzo y orientación con recursos limitados.
#
# ### 1.2 Pregunta de negocio a solucionar
# > **¿Cómo podemos identificar, antes del examen y solo con información del estudiante, su hogar y su colegio, a los
# > estudiantes con alta y baja probabilidad de alcanzar un puntaje global ≥ 300, para focalizar los programas de
# > acompañamiento?**
#
# - **Modelo predictivo.** Variable de salida: `alto_desempeno` (1 si `punt_global ≥ 300`; 0 en otro caso). El modelo
#   entrega la probabilidad de alto desempeño de cada estudiante.
# - **Modelo prescriptivo.** Variable de salida: la **banda de acompañamiento** de cada estudiante, obtenida al aplicar la
#   regla de negocio (1.3) a la probabilidad predicha.
#
# ### 1.3 Regla de negocio
# **Regla 1 — alto desempeño.** Un estudiante tiene alto desempeño si su puntaje global es **≥ 300**: el 60 % de la escala,
# cerca del 17 % superior de Antioquia (se verifica en 2.3).
#
# **Regla 2 — bandas de acompañamiento.** Con el umbral de decisión *t* del modelo final (sección 4.3):
#
# | Banda | Condición sobre la probabilidad *p* | Acción sugerida |
# |---|---|---|
# | 🔴 **Prioridad de nivelación** | p < t/2 | Nivelación en competencias básicas y acompañamiento |
# | 🟠 **Refuerzo focalizado** | t/2 ≤ p < t | Refuerzo en áreas débiles y preparación para la prueba |
# | 🟢 **Alto potencial** | p ≥ t | Estímulos, becas y orientación hacia la educación superior |
#
# ### 1.4 Definición de las funciones del modelo predictivo y prescriptivo
# **Función del modelo predictivo.** Estima, con aprendizaje supervisado, la probabilidad de que un estudiante obtenga
# un puntaje ≥ 300 a partir de 19 variables disponibles **antes** del examen. Se comparan cinco modelos clásicos y tres de
# ensamble, se ajustan sus hiperparámetros y su umbral de decisión con validación cruzada, y el mejor se elige con pruebas
# estadísticas.
#
# **Función del modelo prescriptivo.** Traduce cada probabilidad en una **banda de acompañamiento** (regla 2) y resume las
# bandas por naturaleza del colegio, zona, subregión, jornada y municipio. Así la predicción se convierte en decisiones de
# focalización.
#
# ### 1.5 Definición del criterio de negocio para evaluar el resultado
# El modelo se considera **útil para focalizar** si cumple los tres criterios en el conjunto de prueba (se verifican en 5.5):
#
# | Criterio | Meta | Razón |
# |---|---|---|
# | Detectar al menos la mitad de los estudiantes de alto desempeño | **Recall ≥ 0,55** | Un programa de becas no debe dejar por fuera a la mayoría de los candidatos |
# | Focalizar mejor que el azar | **Lift = Precision / tasa base ≥ 2** | Entre los señalados, la proporción de alto desempeño debe duplicar la de una selección aleatoria |
# | Superar la línea base | **F1 > F1 de la regresión logística base** | El proceso de ajuste debe aportar valor frente al modelo más simple |
#
# ### 1.6 Objetivos de minería de datos y diseño de la solución
# 1. Construir un clasificador con el mayor **F1 de la clase positiva** posible, validado con CV estratificada sobre el 70 %.
# 2. Comparar 5 modelos clásicos (Regresión Logística, KNN, Árbol, SVM, Random Forest) y 3 ensambles (Votación, Bagging, XGBoost).
# 3. Explicar el modelo (coeficientes, importancia por permutación) y auditar su desempeño por grupos.
# 4. Desplegar el pipeline en una aplicación web y aplicar la regla prescriptiva a estudiantes nunca vistos.
#
# | Tipo de análisis | Tipo de aprendizaje | Modelos propuestos | Métrica principal |
# |---|---|---|---|
# | Predictivo + prescriptivo | Supervisado (clasificación binaria) | Logística, KNN, Árbol, SVM (RBF), Random Forest · Votación, Bagging, XGBoost | **F1** de la clase positiva |
#
# **¿Por qué F1?** Con ~17 % de positivos, un modelo que siempre dijera "no alto" acertaría el 83 % de las veces sin ninguna
# utilidad. F1 exige a la vez **precisión** (no señalar de más) y **exhaustividad** (no dejar de detectar).
