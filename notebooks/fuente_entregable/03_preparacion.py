# %% [markdown]
# ## 3. Preparación de los datos
# ### 3.1 Selección de variables

# %% [markdown]
# **¿Qué hacemos y por qué?** Armamos el conjunto de trabajo con las 17 variables originales elegidas por criterio de
# dominio y construimos dos variables nuevas: `edad` y `cole_valle_aburra`. Cada variable queda justificada en una tabla.

# %%
#@copiar 3.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra las **19 variables de entrada** elegidas, cada una con su justificación de dominio: 1 numérica (`edad`), 5 ordinales del hogar y 13 nominales del estudiante, el hogar y el colegio.
# En general, las dos variables **construidas** muestran el valor de la ingeniería de variables:
# - `edad` convierte una fecha de texto en una cantidad con sentido educativo (la extraedad refleja rezago escolar).
# - `cole_valle_aburra` resume 125 municipios (alta cardinalidad) en un indicador binario fácil de interpretar.

# %% [markdown]
# ### 3.2 Descripción estadística

# %% [markdown]
# **¿Qué hacemos y por qué?** Resumimos el conjunto seleccionado antes de limpiarlo: número de registros y variables y
# estadísticos de cada columna. Así queda evidencia del estado original.

# %%
#@copiar 3.2.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra la descripción estadística del conjunto seleccionado: 144.766 registros y 20 variables (las 19 de entrada más el puntaje global), 2 numéricas y 18 categóricas.
# En general, las categorías más frecuentes son estrato 2, madre con bachillerato completo, colegio oficial y jornada de la mañana. La edad tiene media 17,5 años, pero un **mínimo de 0 y un máximo de 79**, evidencia directa de fechas erróneas. Los datos todavía no se han limpiado a propósito: el perfilamiento debe documentar el estado original.

# %% [markdown]
# **¿Qué hacemos y por qué?** Generamos el reporte automático de **ydata-profiling**, entregable exigido por la actividad, con
# distribuciones, faltantes, alertas y correlaciones de cada variable.

# %%
#@copiar 3.2.2

# %% [markdown]
# 🔎 **Interpretación:** La salida confirma que se generó el reporte `reports/perfilamiento_saber11_antioquia.html` (unos 1,8 MB), con histogramas, faltantes, cardinalidad y alertas por variable, además de las matrices de correlación.
# En general, el perfilamiento automático complementa, y no reemplaza, la revisión manual de la fase 2: sus alertas de desbalance, faltantes y alta cardinalidad coinciden con lo encontrado allí. El aviso de deprecación indica que ydata-profiling dejará de actualizarse en favor de `fg-data-profiling`; se mantiene porque la actividad lo exige explícitamente.

# %% [markdown]
# ### 3.3 Limpieza de errores

# %% [markdown]
# **¿Qué hacemos y por qué?** Aplicamos reglas fijas que no aprenden de los datos: eliminar duplicados exactos y resultados
# no oficiales, marcar edades imposibles y validar categorías. Por eso pueden aplicarse antes de partir los datos.

# %%
#@copiar 3.3

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra, regla por regla, cuántos registros afectó la limpieza de errores.
# En general, el conjunto pasa de 144.766 a **72.357 registros**:
# - Se eliminan 72.383 duplicados exactos y 26 resultados no oficiales.
# - Ningún puntaje estaba nulo ni fuera de rango.
# - 1.126 edades imposibles pasan a faltante (la mitad de las 2.252 detectadas, por los duplicados).
# - Ninguna categoría estaba fuera de dominio.
#
# Son reglas **determinísticas**: cada una da el mismo resultado para un registro sin mirar los demás, así que se pueden aplicar antes de partir los datos sin riesgo de fuga de información. La tabla corresponde a la tabla 3.3 del informe.

# %% [markdown]
# **¿Qué hacemos y por qué?** Verificamos con aserciones automáticas que después de limpiar se cumplen todas las reglas de
# calidad de 2.2.

# %%
#@copiar 3.3.1

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra que todas las aserciones pasan: no quedan duplicados, solo hay resultados oficiales, el objetivo es válido en todas las filas, las edades son plausibles y las categorías están dentro de su dominio.
# En general, convertir las reglas de calidad en `assert` es una práctica de **pruebas de datos**: si la fuente cambiara y reapareciera un problema, el cuaderno se detendría aquí en lugar de entrenar un modelo con datos defectuosos.

# %% [markdown]
# ### 3.4 Limpieza de nulos

# %% [markdown]
# **¿Qué hacemos y por qué?** Convertimos "No sabe" y "No Aplica" en faltantes y medimos el % de nulos por variable. La
# imputación se hará dentro del pipeline, para que aprenda solo del entrenamiento.

# %%
#@copiar 3.4

# %% [markdown]
# 🔎 **Interpretación:** El gráfico y la tabla muestran el porcentaje de nulos por variable después de convertir "No sabe" y "No Aplica" en faltantes, junto con el método de imputación de cada una. Los mayores son:
#
# | Variable | % nulos |
# |---|---|
# | `cole_bilingue` | 16,4 |
# | `fami_educacionpadre` | 15,9 (subió desde ~6 % por las respuestas "No sabe") |
# | `fami_educacionmadre` | 8,4 |
# | `fami_estratovivienda` | 6,5 |
# | Tenencias del hogar | entre 4,7 y 5,9 |
# | Edad | 1,6 |
#
# En general, ninguna variable supera el 20 %, así que se **imputan** en lugar de descartarse. La imputación ocurre dentro del pipeline, aprendida solo con los datos de entrenamiento, para no filtrar información de la prueba.

# %% [markdown]
# **¿Qué hacemos y por qué?** Comprobamos si el faltante es **informativo**: si quienes no tienen dato difieren en su tasa de
# alto desempeño, no conviene eliminarlos.

# %%
#@copiar 3.4.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra la tasa de alto desempeño de los registros con y sin dato en cada variable.
# En general, la diferencia es contundente:
# - Quienes **no tienen dato de edad** alcanzan alto desempeño solo en el 0,8 % de los casos, frente al 17,4 % de quienes sí lo tienen.
# - Sin estrato reportado la tasa es 10,3 % (frente a 17,6 %); sin dato de internet, 9,9 % (frente a 17,6 %).
#
# Los faltantes **no son completamente aleatorios** (no son MCAR): estar incompleto se asocia a un perfil de menor desempeño. Eliminar esas filas sesgaría la muestra hacia estudiantes más favorecidos; la categoría `DESCONOCIDO` y los indicadores de faltante permiten que el modelo aproveche esta señal.

# %% [markdown]
# **¿Qué hacemos y por qué?** Guardamos el conjunto limpio, con un identificador que permite rastrear cada fila hasta la
# descarga original.

# %%
#@copiar 3.4.2

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra que se guardó el conjunto limpio (72.357 registros × 22 columnas) con una tasa de alto desempeño del 17,13 %, prácticamente la misma que antes de limpiar.
# En general, la columna `id_registro` conserva la posición original de cada fila, de modo que cualquier registro puede rastrearse hasta la descarga de la API (**trazabilidad**).

# %% [markdown]
# ### 3.5 Codificación de variables categóricas

# %% [markdown]
# **¿Qué hacemos y por qué?** Antes de codificar separamos la muestra de modelado (30.000) y la partición **70/30
# estratificada**. El 30 % de prueba no se toca hasta la fase 5 y el resto queda como *pool* de datos nunca vistos.

# %%
#@copiar 3.5.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el tamaño y la tasa de positivos de cada conjunto:
#
# | Conjunto | Registros |
# |---|---|
# | Muestra de modelado | 30.000 |
# | Entrenamiento | 21.000 |
# | Prueba | 9.000 |
# | Pool que no participa en el modelado | 42.357 |
#
# En general, la estratificación funciona: la tasa de positivos es 17,13 % en todos los conjuntos. La prueba queda reservada hasta la fase 5 y el pool servirá en la fase 6 como datos reales nunca vistos.

# %% [markdown]
# **¿Qué hacemos y por qué?** Definimos el preprocesamiento por tipo de variable:
# - **Numérica:** mediana y estandarización.
# - **Ordinal:** orden explícito, indicador de faltante y estandarización.
# - **Nominal:** categoría DESCONOCIDO y One-Hot.

# %%
#@copiar 3.5.2

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra la estrategia de codificación por tipo de variable (tabla 3.5 del informe):
# - **Edad:** imputación con la mediana y estandarización.
# - **5 ordinales:** código ordinal con orden explícito (p. ej. "Sin Estrato" < "Estrato 1" < … < "Estrato 6"), indicador de faltante y estandarización.
# - **13 nominales:** One-Hot con la categoría `DESCONOCIDO` para los faltantes.
#
# En general, el `ColumnTransformer` todavía no se ha ajustado: es una "receta" que cada pipeline aprenderá **solo con sus datos de entrenamiento**.

# %% [markdown]
# **¿Qué hacemos y por qué?** Ajustamos el preprocesamiento **solo con entrenamiento** y observamos la matriz codificada que
# reciben los modelos.

# %%
#@copiar 3.5.3

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra las primeras filas del entrenamiento ya codificado: las 19 variables se convierten en **50 columnas numéricas** (1 de edad, 5 ordinales con sus 5 indicadores de faltante y 39 columnas one-hot).
# En general, las ordinales y los indicadores quedan estandarizados (media 0) y las nominales quedan como 0/1. Los nombres (`ord__…`, `nom__cole_jornada_MAÑANA`) permiten rastrear cada columna hasta su variable original.

# %% [markdown]
# ### 3.6 Relaciones lineales y no lineales, y reducción de dimensiones

# %% [markdown]
# **¿Qué hacemos y por qué?** Comparamos Pearson (relación lineal) y Spearman (relación monótona) entre las variables
# numéricas u ordinales y el puntaje global, usando solo el entrenamiento.

# %%
#@copiar 3.6.1

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra las matrices de correlación de Pearson y de Spearman entre el puntaje global, la edad y las variables ordinales.
# En general:
# - **Educación de los padres:** es la asociación más fuerte con el puntaje (madre r = 0,39 y ρ = 0,37; padre r = 0,37 y ρ = 0,35).
# - **Edad:** relación negativa (r = −0,28, ρ = −0,30): a mayor edad, menor puntaje.
# - **Estrato:** asociación positiva moderada (0,26).
# - **Multicolinealidad:** las dos educaciones están muy correlacionadas entre sí (0,60), igual que el número de personas y de cuartos del hogar (0,47).
# - **Forma de la relación:** Pearson y Spearman son muy parecidos, así que las relaciones son **aproximadamente monótonas**.
#
# Son asociaciones, no causas: el estrato y la educación de los padres comparten un factor común, el nivel socioeconómico del hogar.

# %% [markdown]
# **¿Qué hacemos y por qué?** Medimos la asociación de cada variable con el objetivo con la **V de Cramér** (categóricas) y
# la **información mutua**, que capta cualquier tipo de dependencia, también la no lineal.

# %%
#@copiar 3.6.2

# %% [markdown]
# 🔎 **Interpretación:** El gráfico y la tabla muestran, para cada variable, la V de Cramér, el valor p de la prueba χ² y la información mutua con el objetivo.
# En general, todas las variables tienen asociación estadísticamente significativa (p < 0,001), algo esperable con n = 21.000, así que lo relevante es el **tamaño** del efecto:
# - Las más asociadas son la educación de la madre (V = 0,30; IM = 0,043) y la del padre (V = 0,29), la edad, tener computador, el estrato y la jornada.
# - Las menos asociadas son la sede principal, el género del estudiante y el número de cuartos (V < 0,08).
#
# La información está repartida entre varias variables y es moderada, lo que anticipa un techo de desempeño modesto para cualquier modelo.

# %% [markdown]
# **¿Qué hacemos y por qué?** Buscamos evidencia visual de no linealidad: el puntaje medio por edad y la tasa de alto
# desempeño por estrato.

# %%
#@copiar 3.6.3

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra el puntaje medio según la edad (izquierda) y la tasa de alto desempeño según el estrato (derecha).
# En general:
# - **Edad:** la relación **no es lineal**. El puntaje medio cae con fuerza entre los 15 y los 20 años (de ~263 a ~210) y luego se estabiliza alrededor de 200–210. Un coeficiente lineal único no describe bien esa forma, y por eso la información mutua sí la capta.
# - **Estrato:** la tasa crece de forma monótona hasta el estrato 5 (~46 %) y baja un poco en el estrato 6 (~42 %, con pocos casos). "Sin estrato" es el grupo más bajo (~5 %).
#
# Por eso tiene sentido comparar modelos lineales con modelos flexibles (árboles, SVM RBF) en la fase 4.

# %% [markdown]
# **¿Qué hacemos y por qué?** Evaluamos la **reducción de dimensiones con PCA**: cuántas componentes hacen falta para
# explicar el 90 % de la varianza de la matriz codificada.

# %%
#@copiar 3.7.1 como 3.6.4

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la varianza explicada acumulada por los componentes principales de las 50 columnas codificadas.
# En general, PCA necesita **17 componentes para explicar el 90 %** de la varianza y 21 para el 95 %, y la curva crece de forma gradual, sin un "codo" marcado. La varianza está repartida porque muchas columnas son dummies casi independientes; reducir a 17 componentes quitaría dimensiones, pero cada componente mezclaría variables heterogéneas (jornada, estrato, tenencias) y perdería su significado.

# %% [markdown]
# **¿Qué hacemos y por qué?** Decidimos con evidencia: comparamos el F1 en validación cruzada de un mismo modelo con y sin PCA.

# %%
#@copiar 3.7.2 como 3.6.5

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el F1 en validación cruzada de la regresión logística con y sin PCA.
# En general, con PCA el F1 **baja** de 0,4685 a 0,4592. Como no mejora (el criterio exigía al menos +0,01), **no se aplica** reducción de dimensiones. PCA maximiza la varianza, no el poder predictivo, y aquí descarta parte de la señal útil para la clase minoritaria. Conservar las variables originales, además, permite interpretar el modelo en la fase 5.

# %% [markdown]
# ### 3.7 Balanceo de clases

# %% [markdown]
# **¿Qué hacemos y por qué?** Con un modelo de referencia comparamos cuatro estrategias frente al desbalance (~17 % de
# positivos): ninguna, ponderación de clases, submuestreo y SMOTE.

# %%
#@copiar 3.8.1 como 3.7.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el efecto de cada estrategia de balanceo sobre la regresión logística en validación cruzada.
# En general, sin balanceo el modelo tiene Precision 0,61 pero **Recall 0,21**: detecta solo uno de cada cinco estudiantes de alto desempeño, y su F1 es 0,316. Las tres estrategias elevan el Recall a ~0,75 y el F1 a ~0,47:
#
# | Estrategia | F1 en CV |
# |---|---|
# | RandomUnderSampler | 0,4696 |
# | class_weight | 0,4685 |
# | SMOTE | 0,4673 |
#
# La **ROC-AUC casi no cambia** (~0,80): el balanceo no mejora la capacidad de ordenar a los estudiantes, solo desplaza el punto de decisión hacia la clase minoritaria. Balancear es, en buena medida, elegir un umbral distinto.

# %% [markdown]
# **¿Qué hacemos y por qué?** Elegimos el muestreador con mejor F1 y visualizamos cómo cambia la distribución de clases del
# entrenamiento. Solo se aplica dentro de los folds de entrenamiento.

# %%
#@copiar 3.8.2 como 3.7.2

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la distribución de clases en un fold de entrenamiento antes y después de aplicar el balanceador elegido, **RandomUnderSampler**.
# En general, el submuestreo deja el entrenamiento balanceado 1:1 (≈3.600 casos por clase, frente a ≈17.400 negativos originales). Su ventaja sobre SMOTE (0,0023 de F1) es mucho menor que la desviación entre folds (~0,012), así que en la práctica las estrategias empatan; se prefiere porque reduce el costo de cómputo (el SVM entrena con ~6 mil filas por fold en lugar de ~28 mil con SMOTE). Solo actúa en `fit` dentro de cada fold, de modo que validación y prueba conservan la distribución real.
