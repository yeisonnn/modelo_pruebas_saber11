# %% [markdown]
# ## 4. Modelamiento
#
# **Variables de entrada y salida del modelo**
#
# | Rol | Variables |
# |---|---|
# | Entrada numérica (1) | `edad` |
# | Entrada ordinal (5) | `fami_estratovivienda`, `fami_educacionmadre`, `fami_educacionpadre`, `fami_personashogar`, `fami_cuartoshogar` |
# | Entrada nominal (13) | `estu_genero`, `fami_tieneinternet`, `fami_tienecomputador`, `fami_tieneautomovil`, `fami_tienelavadora`, `cole_naturaleza`, `cole_jornada`, `cole_area_ubicacion`, `cole_bilingue`, `cole_caracter`, `cole_genero`, `cole_sede_principal`, `cole_valle_aburra` |
# | **Salida** | `alto_desempeno` (1 = puntaje global ≥ 300) |

# %% [markdown]
# **¿Qué hacemos y por qué?** Creamos una "fábrica" de pipelines con preprocesamiento, balanceo y modelo, y una función de
# validación cruzada común. Todos los modelos se evalúan igual y con los mismos 5 folds, así que la comparación es justa y
# sin fuga de información.

# %%
#@copiar 4.0

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra la estructura del pipeline que comparten todos los modelos: `preprocesamiento` → `balanceo` → `modelo`. No hay paso `pca`, por la decisión tomada en 3.6.5.
# En general, con esta fábrica cada modelo se evalúa exactamente igual: mismos folds, mismo preprocesamiento aprendido dentro de cada fold y mismo muestreador. Cualquier diferencia de F1 se debe al algoritmo y no a cómo se prepararon los datos.

# %% [markdown]
# ### 4.1 Modelos clásicos

# %% [markdown]
# **¿Qué hacemos y por qué?** Evaluamos con validación cruzada los cinco modelos clásicos del curso: Regresión Logística,
# KNN, Árbol de Decisión, SVM y Random Forest, con hiperparámetros iniciales razonables.

# %%
#@copiar 4.1.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el desempeño de los cinco modelos clásicos en validación cruzada, ordenados por F1:
#
# | Modelo | F1 en CV |
# |---|---|
# | Regresión Logística | 0,4696 ± 0,012 |
# | Random Forest | 0,4690 ± 0,010 |
# | SVM RBF | 0,4659 ± 0,013 |
# | Árbol de decisión | 0,4298 |
# | KNN | 0,4289 |
#
# En general:
# - **Los tres primeros empatan en la práctica:** sus diferencias son mucho menores que la desviación entre folds y los tres tienen ROC-AUC cercana a 0,80. Que el modelo **lineal** empate con los flexibles indica que la señal disponible es esencialmente aditiva.
# - **El árbol y KNN quedan claramente atrás:** el árbol individual tiene alta varianza y KNN pierde eficacia con 50 dimensiones, en su mayoría binarias.
# - **Costo:** el SVM es, con diferencia, el más lento (unos 25 s frente a 2 s de la logística), aun con el submuestreo.

# %% [markdown]
# **¿Qué hacemos y por qué?** Miramos la variabilidad del F1 entre folds: dos modelos con cajas solapadas pueden no ser
# realmente distintos.

# %%
#@copiar 4.1.2

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la distribución del F1 en los 5 folds de cada modelo clásico.
# En general, las cajas de la logística, Random Forest y el SVM se solapan casi por completo, mientras las del árbol y KNN están claramente más abajo. Mirar la distribución por fold, y no solo la media, evita conclusiones apresuradas: con desviaciones de ~0,01, una diferencia de medias de 0,0006 (logística frente a Random Forest) es ruido. Esa intuición se formaliza con el t-test corregido en 5.2.

# %% [markdown]
# ### 4.2 Modelos de ensamble

# %% [markdown]
# **¿Qué hacemos y por qué?** Implementamos los tres tipos de ensamble que pide la actividad:
# - **Votación suave:** combina los 3 mejores clásicos.
# - **Bagging:** muchos árboles entrenados sobre muestras *bootstrap*.
# - **Boosting (XGBoost):** árboles secuenciales que corrigen el error de los anteriores.
#
# Luego medimos su mejora frente al mejor clásico.

# %%
#@copiar 4.2.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el desempeño de los tres ensambles en validación cruzada. Ninguno supera a la regresión logística:
#
# | Ensamble | F1 en CV | Diferencia con la logística |
# |---|---|---|
# | Votación suave (logística + Random Forest + árbol) | 0,4643 | −0,005 |
# | XGBoost | 0,4606 | −0,009 |
# | Bagging de árboles | 0,4363 | −0,033 |
#
# En general:
# - La **votación** no mejora porque sus miembros cometen errores muy parecidos (usan la misma información) y el árbol débil resta.
# - El **boosting** reduce sesgo, pero aquí el límite no está en el algoritmo, sino en la información que contienen las variables.
# - El **bagging** reduce la varianza del árbol (0,4298 → 0,4363), pero no alcanza a Random Forest, que además decorrelaciona los árboles.
#
# Los ensambles ayudan cuando los modelos base se equivocan de forma distinta o cuando el problema tiene alta varianza; no son una garantía de mejora.

# %% [markdown]
# **¿Qué hacemos y por qué?** Unimos clásicos y ensambles en un único ranking por F1 en validación cruzada, con barras de error.

# %%
#@copiar 4.2.2

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra el ranking conjunto de los 8 modelos, con barras de error (desviación entre folds).
# En general, hay un **grupo de cabeza** de cinco modelos entre 0,46 y 0,47 (logística, Random Forest, SVM, votación y XGBoost) que no se distinguen entre sí; detrás quedan el bagging, el árbol y KNN. Los tres mejores (logística, Random Forest y SVM) pasan al ajuste de hiperparámetros. Cuando varios modelos empatan conviene preferir el más **simple e interpretable**, idea que se retoma en la selección final (5.2).

# %% [markdown]
# ### 4.3 Ajuste de hiperparámetros

# %% [markdown]
# **¿Qué hacemos y por qué?** Definimos, para cada modelo, qué hiperparámetros ajustar, **qué controla** cada uno y **por
# qué** se eligió su rango de búsqueda.

# %%
#@copiar 4.3.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra los espacios de búsqueda de los 8 modelos, aunque solo se ajustan los 3 mejores del ranking.
# En general, cada tipo de hiperparámetro usa una distribución adecuada:
# - **Log-uniforme** para `C` y `gamma`, donde importa el orden de magnitud.
# - **Enteros** para `n_estimators` y `max_depth`.
# - **Listas** para opciones discretas.
#
# No aparece `k_neighbors` de SMOTE porque el balanceador elegido fue el submuestreo, que no tiene ese hiperparámetro; el código lo agrega solo si corresponde.

# %% [markdown]
# **¿Qué hacemos y por qué?** Buscamos la mejor combinación para los 3 mejores modelos, con búsqueda aleatoria y validación
# cruzada 5-fold sobre el 70 % de entrenamiento y F1 como métrica.

# %%
#@copiar 4.3.2

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el resultado de la búsqueda aleatoria (20 combinaciones × 5 folds por modelo):
# - **Regresión Logística:** mejor `C = 0,144` (algo más de regularización que el valor por defecto, 1,0). F1 en CV = 0,4701, en unos 30 s.
# - **Random Forest:** 341 árboles, `max_depth = 20`, `min_samples_leaf = 14`, `max_features = 'sqrt'`. F1 = 0,4696, en 2–4 minutos.
# - **SVM:** `C = 0,67` y `gamma = 0,138`. F1 = 0,4655, en unos 15–20 minutos: fue, con diferencia, la búsqueda más costosa. (Los tiempos exactos aparecen en la salida y varían entre ejecuciones; el orden de magnitud es estable.)
#
# En general, las mejoras frente a los valores por defecto son mínimas (≤ 0,0006): los hiperparámetros por defecto ya estaban cerca del óptimo y **el límite está en los datos, no en la configuración**. Para el SVM se usa búsqueda aleatoria y no por mitades sucesivas, porque las primeras rondas de esta última usan tan pocos datos que el F1 no alcanza a distinguir configuraciones.

# %% [markdown]
# **¿Qué hacemos y por qué?** Documentamos cada hiperparámetro ajustado en una tabla: qué controla, rango, razón del rango,
# valor final y **efecto observado** en el F1.

# %%
#@copiar 4.3.3

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra, por hiperparámetro, qué controla, el rango explorado, la razón del rango, el valor final y el efecto observado (tabla 4.3 del informe).
# En general:
# - **`C` de la logística:** aumentarlo mejoró el F1 (ρ = +0,63, p = 0,003). Los valores muy pequeños (regularización excesiva) empeoran el modelo y, desde ~0,1, el F1 se estabiliza.
# - **`C` del SVM:** también tiene un efecto positivo significativo (ρ = +0,69).
# - **Random Forest:** ninguno de sus hiperparámetros tiene efecto significativo en los rangos probados (p > 0,4); es un modelo robusto a su configuración.
# - **`gamma` del SVM:** su efecto no es significativo en el rango (p = 0,125).

# %% [markdown]
# **¿Qué hacemos y por qué?** Visualizamos cómo cambia el F1 de validación con cada hiperparámetro.

# %%
#@copiar 4.3.4

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra el F1 de cada combinación probada frente al valor de cada hiperparámetro.
# En general:
# - **Logística:** el F1 sube rápido cuando `C` se aleja de valores cercanos a 0 y luego forma una **meseta** (~0,469–0,470). Más allá de C ≈ 0,1 la regularización deja de importar.
# - **Random Forest:** nubes de puntos sin tendencia, coherentes con la ausencia de efecto significativo.
# - **SVM:** los valores de `C` muy pequeños dan los peores F1.
#
# Ninguna curva sugiere que el óptimo esté fuera de los rangos probados, así que no hace falta ampliarlos.

# %% [markdown]
# **¿Qué hacemos y por qué?** Ajustamos el **umbral de decisión** que maximiza el F1 en validación cruzada: el valor de *t*
# que usa la regla de negocio 2. Luego medimos la mejora total frente a los modelos base.

# %%
#@copiar 4.3.5

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el F1 en validación cruzada antes y después de ajustar el **umbral de decisión**:
#
# | Modelo | F1 en CV | Umbral óptimo |
# |---|---|---|
# | Regresión Logística | 0,4701 → 0,4856 | 0,615 |
# | Random Forest | 0,4696 → 0,4815 | 0,580 |
# | SVM | 0,4655 → 0,4748 | 0,432 |
#
# En general, es la mayor mejora de toda la fase. El submuestreo deja al modelo "creyendo" que la mitad de los estudiantes son positivos, así que sus probabilidades quedan infladas; subir el umbral por encima de 0,5 corrige ese sesgo y equilibra Precision y Recall. En problemas desbalanceados, el umbral es un hiperparámetro más y suele importar más que la configuración del algoritmo.
