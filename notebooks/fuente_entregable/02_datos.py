# %% [markdown]
# ## 2. Entendimiento de los datos
# ### 2.0 Carga de la información

# %% [markdown]
# **¿Qué hacemos y por qué?** Descargamos de la API de Datos Abiertos Colombia solo los registros de Antioquia del periodo
# 2022-2, usando un filtro SoQL en el servidor, y guardamos una copia local comprimida. Si la copia ya existe, se lee de disco.

# %%
#@copiar 2.0.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra, en vista transpuesta, los primeros registros de la copia local del CSV descargado de la API: **144.766 filas × 51 columnas**. La primera vez los datos se descargan de datos.gov.co con el filtro SoQL `periodo='20224' AND cole_depto_ubicacion='ANTIOQUIA'`.
# En general, cada registro mezcla códigos DANE (`05034`), categorías socioeconómicas del hogar y puntajes. Leer todo como texto (`dtype=str`) evita que los códigos pierdan el cero inicial; los puntajes se convierten a número en la celda siguiente.

# %% [markdown]
# **¿Qué hacemos y por qué?** Convertimos los puntajes a número y revisamos, columna por columna, el tipo, los nulos y la
# cardinalidad. Es la "revisión inicial" que orienta la limpieza.

# %%
#@copiar 2.0.2

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra, por columna, el tipo de dato, el porcentaje de nulos y la cardinalidad (valores únicos) tras convertir los seis puntajes a numérico.
# En general, el resumen ya revela los primeros problemas de calidad:
# - `estu_consecutivo` tiene solo **72.383 valores únicos en 144.766 filas**: cada evaluado aparece dos veces (se confirma en 2.2.2).
# - Hay nulos entre 3 % y 16 % en variables del hogar y del colegio (`cole_bilingue` llega al 16,4 %).
# - `periodo`, `cole_depto_ubicacion` y `estu_estudiante` tienen un único valor: son constantes por construcción del filtro.
#
# La cardinalidad anticipa qué columnas son identificadores (códigos de colegio con más de 1.000 valores) y cuáles son categóricas útiles (entre 2 y 12 valores).

# %% [markdown]
# ### 2.1 Diccionario de datos
# **Descripción de las variables que se usarán en el modelo:**
# - **edad:** edad en años al examen (derivada de la fecha de nacimiento). Cuantitativa discreta.
# - **estu_genero:** género del estudiante (F/M). Cualitativa nominal.
# - **fami_estratovivienda:** estrato de la vivienda (Sin estrato, 1 a 6). Cualitativa ordinal.
# - **fami_educacionmadre / fami_educacionpadre:** máximo nivel educativo (de Ninguno a Postgrado). Cualitativas ordinales.
# - **fami_personashogar:** personas en el hogar (1–2 a 9 o más). Cualitativa ordinal.
# - **fami_cuartoshogar:** cuartos del hogar (Uno a Seis o más). Cualitativa ordinal.
# - **fami_tieneinternet / tienecomputador / tieneautomovil / tienelavadora:** tenencia en el hogar (Si/No). Cualitativas nominales.
# - **cole_naturaleza:** oficial o no oficial. Cualitativa nominal.
# - **cole_jornada:** mañana, tarde, completa, única, noche o sabatina. Cualitativa nominal.
# - **cole_area_ubicacion:** urbano o rural. Cualitativa nominal.
# - **cole_bilingue / cole_sede_principal:** S/N. Cualitativas nominales.
# - **cole_caracter:** académico, técnico, técnico/académico o no aplica. Cualitativa nominal.
# - **cole_genero:** mixto, femenino o masculino. Cualitativa nominal.
# - **cole_valle_aburra:** colegio en el área metropolitana (derivada del municipio). Cualitativa nominal.
#
# **Variable de salida:** `alto_desempeno`, binaria. Vale 1 cuando el **puntaje global** (0–500) es **≥ 300** y 0 en otro
# caso. La tabla siguiente documenta además las 51 columnas originales, con su rol y el motivo de exclusión de las que no se usan.

# %% [markdown]
# **¿Qué hacemos y por qué?** Construimos el diccionario completo: descripción, tipo (Num/Cat), rol (entrada, salida o
# excluida) y motivo de exclusión. Aquí se decide qué columnas serían **fuga de información**.

# %%
#@copiar 2.1.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el diccionario de datos: 54 variables (51 originales, 2 derivadas y el objetivo) con su descripción, tipo (Num/Cat), rol y motivo de exclusión.
# En general:
# - Entran al modelo **19 variables**: 1 numérica, 5 ordinales y 13 nominales.
# - Seis columnas se excluyen por **fuga de información**: los cinco puntajes por área y `desemp_ingles`, que son componentes del puntaje global.
# - El resto se descarta por ser identificadores, constantes o redundantes con la ubicación del colegio.
#
# Esta tabla corresponde a la tabla 2.1 del informe y es el punto de partida de la selección de variables (3.1).

# %% [markdown]
# **¿Qué hacemos y por qué?** Comprobamos con los datos que las columnas descartadas por "constantes" realmente lo son.

# %%
#@copiar 2.1.2

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra, para las columnas candidatas a descartarse, el porcentaje que ocupa su categoría más frecuente.
# En general, confirma lo declarado en el diccionario: `periodo`, `cole_depto_ubicacion` y `estu_estudiante` son 100 % constantes, y `cole_calendario` (99,7 % A), `estu_privado_libertad` (99,85 % N) y la nacionalidad y la residencia (98,5 % Colombia) son casi constantes. Una variable con más del ~98 % en una sola categoría apenas ayuda a distinguir clases y genera columnas one-hot con muy pocos casos, así que excluirla simplifica el modelo sin perder información.

# %% [markdown]
# ### 2.2 Reglas de calidad

# %% [markdown]
# **¿Qué hacemos y por qué?** Definimos qué valores son aceptables: rangos de los puntajes y de la edad, y dominios de cada
# categoría. Luego medimos cuántos registros incumplen cada regla. La limpieza de la fase 3 responderá a estos incumplimientos.

# %%
#@copiar 2.2.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el resultado de las reglas numéricas: los puntajes respetan sus escalas (0 % fuera de rango) y solo `punt_ingles` tiene 102 nulos.
# En general, la regla de **consistencia** de la edad es la que falla: 2.252 filas (1,56 %) tienen edades fuera de [13, 30], producto de fechas de nacimiento mal registradas (en 3.2.1 aparece incluso una edad mínima de 0 años). Esas filas no se eliminan: en 3.3 la edad se marca como faltante, porque el resto del registro es válido.

# %% [markdown]
# **¿Qué hacemos y por qué?** Verificamos los dominios categóricos y la **unicidad** de los registros (duplicados e
# identificadores repetidos).

# %%
#@copiar 2.2.2

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el resultado de las reglas de validez, unicidad y estado del resultado.
# En general:
# - **Validez:** ninguna variable tiene valores fuera de su dominio (0 % de incumplimiento), así que no hacen falta homologaciones de texto.
# - **Unicidad:** hay **72.383 filas duplicadas exactas** (las 51 columnas iguales) y el mismo número de identificadores repetidos. La mitad del archivo publicado es copia de la otra mitad: es un problema de la fuente, no del análisis.
# - **Resultados no oficiales:** 52 filas tienen el estado "VALIDEZ OFICINA JURÍDICA" (26 estudiantes, porque también están duplicadas).
#
# Sin esta regla, el modelo se habría entrenado con datos repetidos y la validación sería optimista, porque el mismo estudiante podría quedar a la vez en entrenamiento y en prueba.

# %% [markdown]
# ### 2.3 Distribución de la variable objetivo

# %% [markdown]
# **¿Qué hacemos y por qué?** Observamos la distribución del puntaje global y cuántos estudiantes superan el umbral de 300.
# Así cuantificamos el **desbalance** entre clases.

# %%
#@copiar 2.3

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la distribución del puntaje global con el umbral de 300 marcado (izquierda) y el tamaño de cada clase de la variable objetivo (derecha).
# En general, el puntaje tiene media 245,9, mediana 242 y desviación 51,6, con una ligera asimetría positiva (0,31): pocos estudiantes alcanzan puntajes muy altos. El umbral de 300 queda en el **percentil 82,9**, así que solo el 17,1 % de los registros es "alto desempeño" (24.794 frente a 119.972, antes de quitar duplicados). Es un problema **desbalanceado**: un modelo que siempre dijera "no alto" acertaría el 82,9 % de las veces sin ninguna utilidad. Por eso la métrica principal es F1 y en 3.7 se evalúa el balanceo.

# %% [markdown]
# ### 2.4 Análisis exploratorio de datos (EDA)

# %% [markdown]
# **¿Qué hacemos y por qué?** Preparamos la base del EDA **sin duplicados**, para que cada estudiante cuente una sola vez, y
# con las variables derivadas: edad válida, subregión e indicador de alto desempeño.

# %%
# 2.4.0 Base para el análisis exploratorio: cada estudiante una sola vez y variables derivadas
df_eda = df_raw.drop_duplicates().copy()                                   # Evita contar dos veces a cada evaluado
df_eda["edad"] = calcular_edad(df_eda["estu_fechanacimiento"])             # Edad en años cumplidos
df_eda.loc[~df_eda["edad"].between(13, 30), "edad"] = np.nan              # Edades imposibles quedan fuera del análisis
df_eda["subregion"] = np.where(df_eda["cole_cod_mcpio_ubicacion"].isin(CODIGOS_VALLE_ABURRA),
                               "Valle de Aburrá", "Resto de Antioquia")     # Área metropolitana frente al resto
df_eda["alto"] = (df_eda["punt_global"] >= UMBRAL_ALTO_DESEMPENO).astype(int)   # 1 = alto desempeño (regla 1)
# Ficha rápida de la base del EDA
print(f"Estudiantes únicos: {len(df_eda):,} | tasa de alto desempeño: {df_eda['alto'].mean():.2%}")
print(f"Municipios: {df_eda['cole_mcpio_ubicacion'].nunique()} | "
      f"establecimientos: {df_eda['cole_cod_dane_establecimiento'].nunique():,} | "
      f"sedes: {df_eda['cole_cod_dane_sede'].nunique():,}")

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra la ficha de la base del análisis exploratorio: **72.383 estudiantes únicos** (una vez quitados los duplicados exactos), con una tasa de alto desempeño del 17,13 %, distribuidos en 125 municipios, 1.122 establecimientos y 1.447 sedes.
# En general, explorar sobre estudiantes únicos evita que los conteos salgan duplicados. Las tres variables derivadas (`edad`, `subregion` y `alto`) se calculan una sola vez aquí y se reutilizan en todos los gráficos de la sección. Esta base es solo descriptiva: la limpieza formal que alimenta el modelo se hace en la fase 3.

# %% [markdown]
# #### 2.4.1 Histogramas
# **¿Qué hacemos y por qué?** Dibujamos el histograma del puntaje global, de los cinco puntajes por área y de la edad, con
# la mediana marcada, para conocer la forma de cada distribución (simetría, colas, concentración). Los puntajes por área
# **no entran al modelo** (son fuga de información), pero ayudan a entender el puntaje global.

# %%
# 2.4.1 Histogramas de los puntajes y de la edad (forma de cada distribución)
COLS_HIST = ["punt_global", "punt_matematicas", "punt_lectura_critica", "punt_sociales_ciudadanas",
             "punt_c_naturales", "punt_ingles", "edad"]
fig, ejes = plt.subplots(2, 4, figsize=(18, 8))                            # Rejilla de 8 paneles (7 variables)
for eje, col in zip(ejes.flat, COLS_HIST):
    serie = df_eda[col].dropna()                                            # Sin nulos para el histograma
    eje.hist(serie, bins=40, color="#3b82f6", edgecolor="black")            # Barras con borde para distinguirlas
    eje.axvline(serie.median(), color="crimson", ls="--", lw=1.5, label=f"mediana = {serie.median():.0f}")
    eje.set_title(col); eje.legend(fontsize=8)
ejes.flat[-1].axis("off")                                                   # El octavo panel queda vacío
plt.suptitle("Distribución de los puntajes y la edad (estudiantes únicos)", fontsize=14)
plt.tight_layout(); guardar_figura("e_histogramas"); plt.show()
# Resumen numérico que acompaña a los histogramas
resumen_numerico = df_eda[COLS_HIST].describe().T.round(2)
guardar_tabla(resumen_numerico.reset_index().rename(columns={"index": "variable"}), "e_resumen_numerico")
resumen_numerico

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra los histogramas de los seis puntajes y de la edad, con la mediana marcada en rojo.
# En general:
# - El **puntaje global** (mediana 242, media 245,9, rango 100–477) tiene forma de campana con una cola larga hacia la derecha: pocos estudiantes superan los 350 puntos.
# - Los **puntajes por área** se concentran entre 40 y 60 (medianas de 47 a 53). Lectura crítica es la más alta (mediana 53) e inglés la más asimétrica, con una acumulación en 100 que marca el tope de la escala.
# - La **edad** se concentra en 16 y 17 años (mediana 17), pero tiene una cola hasta los 30 años que corresponde a estudiantes en extraedad y a programas de adultos.
#
# Los puntajes por área no se usan como entradas del modelo, porque forman el puntaje global (fuga de información); aquí solo describen la prueba.

# %% [markdown]
# #### 2.4.2 Frecuencias de las variables categóricas
# **¿Qué hacemos y por qué?** Para cada variable categórica de entrada mostramos el porcentaje de estudiantes en cada
# categoría, con "Sin dato" como una categoría más. Es el equivalente categórico del histograma y revela las categorías
# poco representadas.

# %%
# 2.4.2 Frecuencia (%) de cada categoría en las variables categóricas de entrada
VARS_CAT_EDA = [v for v in FEATURES if v not in ("edad", "cole_valle_aburra")] + ["subregion"]   # 18 variables
fig, ejes = plt.subplots(6, 3, figsize=(18, 26))
for eje, var in zip(ejes.flat, VARS_CAT_EDA):
    pct = df_eda[var].fillna("Sin dato").value_counts(normalize=True).mul(100)   # % por categoría (incluye faltantes)
    if var in VARIABLES_ORDINALES:                                              # Las ordinales se muestran en su orden natural
        pct = pct.reindex([c for c in CATEGORIAS_VALIDAS[var] + ["Sin dato"] if c in pct.index])
    eje.barh(pct.index.astype(str), pct.values, color="#3b82f6")
    for y_pos, valor in enumerate(pct.values):
        eje.text(valor, y_pos, f" {valor:.1f}%", va="center", fontsize=8)     # Etiqueta con el porcentaje
    eje.set_title(var, fontsize=11); eje.invert_yaxis()                        # Primera categoría arriba
plt.tight_layout(); guardar_figura("e_frecuencias_categoricas"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra el porcentaje de estudiantes en cada categoría de las 18 variables categóricas de entrada, incluida la categoría "Sin dato".
# En general:
# - **Hogar:** predominan el estrato 2 (36,8 %) y el 3 (25,7 %), madres y padres con bachillerato completo (27,5 % y 22,3 %) y hogares de 3 a 4 personas (53,4 %). El 73,7 % tiene internet, pero solo el 58,8 % tiene computador.
# - **Colegio:** el 79,6 % estudia en colegios oficiales, el 82,7 % en zona urbana y el 58,9 % en el Valle de Aburrá.
# - **Categorías casi vacías:** colegio bilingüe (0,3 %), colegios masculinos (0,4 %) y estrato 6 (0,9 %). Al codificarlas generan columnas con muy pocos casos.
# - **Faltantes:** "No sabe" es frecuente en la educación del padre (8,2 %) y `cole_bilingue` no tiene dato en el 16,4 % de los casos.
#
# Estas frecuencias guían la codificación (3.5) y el tratamiento de nulos (3.4).

# %% [markdown]
# #### 2.4.3 Gráficos exploratorios frente a la variable objetivo
# **¿Qué hacemos y por qué?** Comparamos la tasa de alto desempeño entre las categorías de cuatro factores clave (educación
# de la madre, estrato, computador en el hogar y jornada) con la tasa global como referencia. Así se ve qué factores
# "separan" a los estudiantes.

# %%
# 2.4.3 Tasa de alto desempeño (%) según factores del hogar y del colegio
FACTORES = {"fami_educacionmadre": "Educación de la madre", "fami_estratovivienda": "Estrato",
            "fami_tienecomputador": "Computador en el hogar", "cole_jornada": "Jornada"}
tasa_global = 100 * df_eda["alto"].mean()                                  # Línea de referencia de cada panel
fig, ejes = plt.subplots(2, 2, figsize=(18, 11))
for eje, (var, titulo) in zip(ejes.flat, FACTORES.items()):
    tasa = df_eda.groupby(df_eda[var].fillna("Sin dato"))["alto"].mean().mul(100)   # % de alto desempeño por categoría
    if var in VARIABLES_ORDINALES:                                          # Orden natural para las ordinales
        tasa = tasa.reindex([c for c in CATEGORIAS_VALIDAS[var] + ["Sin dato"] if c in tasa.index])
    else:                                                                   # De menor a mayor para las nominales
        tasa = tasa.sort_values()
    eje.barh(tasa.index.astype(str), tasa.values, color="#10b981")
    eje.axvline(tasa_global, color="black", ls="--", lw=1, label=f"Tasa global = {tasa_global:.1f}%")
    for y_pos, valor in enumerate(tasa.values):
        eje.text(valor, y_pos, f" {valor:.1f}%", va="center", fontsize=9)    # Valor al final de cada barra
    eje.set_title(f"% de alto desempeño por {titulo.lower()}"); eje.legend(loc="lower right")
plt.tight_layout(); guardar_figura("e_tasa_por_factor"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la tasa de alto desempeño en cada categoría de cuatro factores (educación de la madre, estrato, computador y jornada) frente a la tasa global de 17,1 % (línea punteada).
# En general:
# - **Educación de la madre:** la tasa crece casi en escalera, desde 1,9 % (ninguna) hasta 59,6 % (postgrado). Es el gradiente más marcado de todos.
# - **Estrato:** sube de 7,9 % (estrato 1) a 46,9 % (estrato 5); "Sin estrato" queda en 4,5 %.
# - **Computador:** 23,8 % con computador frente a 6,8 % sin él: más del triple.
# - **Jornada:** completa (26,5 %) y única (20,0 %) superan la media, mientras la nocturna y la sabatina apenas llegan al 2 %.
#
# Los cuatro factores separan bien las clases, así que son candidatos fuertes para el modelo.

# %% [markdown]
# **¿Qué hacemos y por qué?** Con diagramas de caja comparamos la **distribución completa** del puntaje global (mediana,
# cuartiles y atípicos) según naturaleza, zona, subregión y jornada del colegio. La línea roja marca el umbral de 300.

# %%
# 2.4.4 Diagramas de caja del puntaje global según características del colegio
GRUPOS_BOX = {"cole_naturaleza": "Naturaleza", "cole_area_ubicacion": "Zona", "subregion": "Subregión", "cole_jornada": "Jornada"}
fig, ejes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)               # Mismo eje Y para comparar niveles
for eje, (var, titulo) in zip(ejes, GRUPOS_BOX.items()):
    orden = df_eda.groupby(var)["punt_global"].median().sort_values().index   # Cajas ordenadas por mediana
    sns.boxplot(data=df_eda, x=var, y="punt_global", order=orden, ax=eje, color="#93c5fd", fliersize=1)
    eje.axhline(UMBRAL_ALTO_DESEMPENO, color="crimson", ls="--", lw=1)          # Umbral de alto desempeño
    eje.set_title(titulo); eje.set_xlabel(""); eje.tick_params(axis="x", rotation=30)
plt.tight_layout(); guardar_figura("e_boxplots_colegio"); plt.show()
# Mediana del puntaje por grupo: el número que resume cada caja
pd.concat({titulo: df_eda.groupby(var)["punt_global"].median() for var, titulo in GRUPOS_BOX.items()}).rename("mediana").to_frame()

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra diagramas de caja del puntaje global según la naturaleza, la zona, la subregión y la jornada del colegio, con el umbral de 300 en rojo; la tabla da la mediana de cada grupo.
# En general, la mediana es más alta en colegios no oficiales (264 frente a 238), urbanos (247 frente a 219) y del Valle de Aburrá (253 frente a 227). En la jornada el contraste es mayor: la sabatina (199) y la nocturna (205) quedan casi enteras por debajo de 250, mientras la única (257) y la completa (256) llegan con su cuartil superior hasta cerca de 300.
# En todos los grupos las cajas se solapan mucho: hay estudiantes de alto desempeño en todos los contextos. Por eso ninguna variable sola basta para predecir, y el modelo debe combinarlas.

# %% [markdown]
# **¿Qué hacemos y por qué?** Usamos un **gráfico de radar** para comparar de un vistazo los 10 municipios con mayor
# puntaje global promedio. Solo se incluyen municipios con al menos 300 estudiantes, para que el promedio sea estable.

# %%
# 2.4.5 Radar: los 10 municipios con mayor puntaje global promedio (con al menos 300 estudiantes)
por_mcpio = (df_eda.groupby("cole_mcpio_ubicacion")
             .agg(estudiantes=("punt_global", "size"), puntaje_medio=("punt_global", "mean"), tasa_alto=("alto", "mean")))
top10 = por_mcpio[por_mcpio["estudiantes"] >= 300].nlargest(10, "puntaje_medio")   # Umbral de tamaño: promedios estables
etiquetas = top10.index.tolist() + [top10.index[0]]                        # El radar se cierra repitiendo el primer punto
valores = top10["puntaje_medio"].tolist() + [top10["puntaje_medio"].iloc[0]]
angulos = np.linspace(0, 2 * np.pi, len(etiquetas))                        # Un ángulo por municipio (el último = el primero)
fig, eje = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
eje.plot(angulos, valores, color="#1d4ed8", lw=2); eje.fill(angulos, valores, color="#1d4ed8", alpha=0.25)
eje.set_xticks(angulos[:-1]); eje.set_xticklabels(etiquetas[:-1], fontsize=9)
eje.set_ylim(por_mcpio["puntaje_medio"].min() - 10, top10["puntaje_medio"].max() + 5)   # Escala que resalta las diferencias
eje.set_title("Top 10 municipios por puntaje global promedio (≥ 300 estudiantes)", y=1.08, fontsize=13)
plt.tight_layout(); guardar_figura("e_radar_municipios"); plt.show()
top10.round(3)

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra, en forma de radar, el puntaje global promedio de los 10 municipios con mejor resultado entre los que tienen al menos 300 estudiantes.
# En general, Envigado encabeza con 282,5 puntos, seguido de Sabaneta (278,1) y La Estrella (271,0); el décimo municipio ronda los 261. Los municipios del grupo pertenecen al Valle de Aburrá (Envigado, Sabaneta, La Estrella, Itagüí, Copacabana) o al Oriente cercano (La Ceja, Rionegro, Marinilla, El Carmen de Viboral), con Santa Rosa de Osos como excepción del Norte.
# Todos están por encima de la media departamental (245,9), pero ninguno alcanza en promedio el umbral de 300: incluso en los mejores municipios, el alto desempeño es minoritario.

# %% [markdown]
# #### 2.4.6 Tablas exploratorias de datos
# **¿Qué hacemos y por qué?** Listamos los 10 municipios con **mayor** y los 10 con **menor** tasa de alto desempeño (con
# al menos 300 estudiantes). Esta tabla "top N" pone en cifras la brecha territorial del departamento.

# %%
# 2.4.6 Municipios con mayor y menor tasa de alto desempeño (≥ 300 estudiantes)
base_mcpio = (por_mcpio[por_mcpio["estudiantes"] >= 300]
              .assign(tasa_alto_pct=lambda d: (100 * d["tasa_alto"]).round(2),       # Tasa en porcentaje
                      puntaje_medio=lambda d: d["puntaje_medio"].round(1))
              .drop(columns="tasa_alto"))
tabla_top = pd.concat({"Mayor tasa": base_mcpio.nlargest(10, "tasa_alto_pct"),
                       "Menor tasa": base_mcpio.nsmallest(10, "tasa_alto_pct")})     # Dos bloques en una tabla
guardar_tabla(tabla_top.reset_index().rename(columns={"level_0": "grupo", "cole_mcpio_ubicacion": "municipio"}), "e_municipios")
print(f"Municipios con al menos 300 estudiantes: {len(base_mcpio)} de {len(por_mcpio)}")
tabla_top

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra los 10 municipios con mayor y los 10 con menor tasa de alto desempeño, entre los 38 municipios (de 125) con al menos 300 estudiantes.
# En general, la brecha territorial es enorme: en Envigado el 40,1 % de los estudiantes supera los 300 puntos, frente al 1,3 % en Anorí, una diferencia de 30 veces. Los municipios con menor tasa se concentran en el Bajo Cauca y el Nordeste (Anorí, Remedios, Zaragoza, Cáceres, El Bagre, Segovia) y en Urabá (Necoclí, Arboletes, Turbo, San Pedro de Urabá), con puntajes medios entre 192 y 224.
# Esta diferencia justifica incluir la subregión en el modelo y, en la fase 6, analizar las bandas de acompañamiento por municipio.

# %% [markdown]
# **¿Qué hacemos y por qué?** Con una **tabla dinámica** cruzamos estrato y naturaleza del colegio y calculamos la tasa de
# alto desempeño de cada combinación, con fila y columna de **totales**. La tabla de conteos acompaña a la de tasas para
# saber cuántos estudiantes respaldan cada porcentaje.

# %%
# 2.4.7 Tabla dinámica: tasa de alto desempeño (%) por estrato y naturaleza del colegio, con totales
pivote = pd.pivot_table(df_eda, values="alto", index="fami_estratovivienda", columns="cole_naturaleza",
                        aggfunc="mean", margins=True, margins_name="TOTAL").mul(100).round(1)   # margins = fila y columna TOTAL
pivote = pivote.reindex([c for c in CATEGORIAS_VALIDAS["fami_estratovivienda"] + ["TOTAL"] if c in pivote.index])
conteo = pd.crosstab(df_eda["fami_estratovivienda"], df_eda["cole_naturaleza"],
                     margins=True, margins_name="TOTAL").reindex(pivote.index)                 # Estudiantes por celda
guardar_tabla(pivote.reset_index(), "e_pivote_estrato_naturaleza")
print("Estudiantes por celda:"); display(conteo)
print("Tasa de alto desempeño (%):")
pivote

# %% [markdown]
# 🔎 **Interpretación:** La tabla dinámica muestra la tasa de alto desempeño (%) cruzando el estrato con la naturaleza del colegio; la fila y la columna TOTAL dan las tasas marginales. Encima se muestra cuántos estudiantes hay en cada celda (67.661 con estrato reportado; por eso el TOTAL es 17,6 % y no 17,1 %).
# En general:
# - En los **colegios no oficiales** la tasa crece con el estrato de forma sostenida: de 7,0 % en el estrato 1 a 62,3 % en el estrato 5.
# - En los **oficiales** crece solo hasta el estrato 3 (20,2 %) y luego **cae** (15,9 % en el estrato 4, 9,5 % en el 5 y 4,2 % en el 6). Las dos últimas celdas son pequeñas (495 y 216 estudiantes), así que conviene leerlas con cautela.
# - En el estrato 1 ocurre lo contrario: el colegio oficial (8,0 %) supera levemente al no oficial (7,0 %).
#
# El efecto del estrato depende del tipo de colegio: es una **interacción**. Un modelo aditivo como la regresión logística solo la capta de forma aproximada.

# %% [markdown]
# #### 2.4.8 Análisis de correlación
# **¿Qué hacemos y por qué?** Calculamos la correlación de **Spearman** (adecuada para variables ordinales) entre los
# puntajes, la edad y las variables ordinales del hogar codificadas en su orden natural. Después listamos los pares más
# correlacionados, que se comentan uno a uno.

# %%
# 2.4.8 Matriz de correlación de Spearman entre puntajes, edad y variables ordinales del hogar
df_corr = df_eda[COLS_PUNTAJES + ["edad"]].copy()
for var in VARIABLES_ORDINALES:
    orden = [c for c in CATEGORIAS_VALIDAS[var] if c not in ("No sabe", "No Aplica")]      # Respuestas sin orden quedan como NaN
    df_corr[var] = df_eda[var].map({c: i for i, c in enumerate(orden)})                  # Código ordinal 0, 1, 2, ...
matriz = df_corr.corr(method="spearman")
fig, eje = plt.subplots(figsize=(12, 9))
sns.heatmap(matriz, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, linewidths=0.5, ax=eje)
eje.set_title("Matriz de correlación de Spearman (estudiantes únicos)")
plt.tight_layout(); guardar_figura("e_correlaciones"); plt.show()
# Pares ordenados por magnitud (triángulo superior, sin la diagonal): insumo para la explicación
pares = matriz.where(np.triu(np.ones(matriz.shape, dtype=bool), k=1)).stack().rename("rho").reset_index()
pares.reindex(pares["rho"].abs().sort_values(ascending=False).index).head(12).round(3)

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la matriz de correlación de Spearman entre los puntajes, la edad y las variables ordinales del hogar; la tabla lista los 12 pares con mayor correlación en valor absoluto, y todos son pares de puntajes.
# En general:
# - Los **puntajes por área** se correlacionan muy fuerte con el global (ρ = 0,90–0,91; inglés 0,78) y entre sí (0,66–0,81). Por eso no pueden usarse como entradas: serían una fuga de información.
# - Entre las variables de entrada, la **educación de la madre** (ρ = 0,37) y la **del padre** (0,35) son las más asociadas con el puntaje global, seguidas por el **estrato** (0,24). La **edad** tiene relación negativa (−0,30).
# - Hay **multicolinealidad moderada** entre las dos educaciones (0,60) y entre personas y cuartos del hogar (0,46).
#
# Ninguna correlación entre entradas es tan alta como para obligar a eliminar variables, pero sí explica por qué el estrato pierde importancia en el modelo cuando ya están las educaciones (5.3.9).

# %% [markdown]
# ### 2.5 Sesgos en los datos

# %% [markdown]
# **¿Qué hacemos y por qué?** Medimos, para seis variables de grupo, qué porcentaje de estudiantes representa cada grupo
# (**sesgo de representación**) y cuál es su tasa de alto desempeño (**inequidad de base**, o sesgo histórico que el modelo
# podría aprender).

# %%
# 2.5 Representación de cada grupo y su tasa de alto desempeño (estudiantes únicos)
GRUPOS_SESGO = {"estu_genero": "Género del estudiante", "cole_area_ubicacion": "Zona del colegio",
                "cole_naturaleza": "Naturaleza del colegio", "subregion": "Subregión",
                "fami_estratovivienda": "Estrato", "cole_jornada": "Jornada"}
filas = []
for var, titulo in GRUPOS_SESGO.items():
    grupos = df_eda.groupby(df_eda[var].fillna("Sin dato"))["alto"].agg(["size", "mean"])   # Tamaño y tasa por grupo
    for grupo, fila in grupos.iterrows():
        filas.append({"variable": titulo, "grupo": grupo, "estudiantes": int(fila["size"]),
                      "pct_estudiantes": 100 * fila["size"] / len(df_eda), "tasa_alto_desempeno_pct": 100 * fila["mean"]})
tabla_sesgos = guardar_tabla(pd.DataFrame(filas).round(2), "e_sesgos")
fig, ejes = plt.subplots(2, 3, figsize=(17, 8))
for eje, (titulo, sub) in zip(ejes.flat, tabla_sesgos.groupby("variable", sort=False)):   # Un panel por variable
    eje.bar(sub["grupo"].astype(str), sub["tasa_alto_desempeno_pct"], color="#4C72B0")
    eje.axhline(100 * df_eda["alto"].mean(), ls="--", color="k", lw=1)                    # Tasa global como referencia
    eje.set_title(titulo); eje.set_ylabel("% alto desempeño"); eje.tick_params(axis="x", rotation=35)
plt.tight_layout(); guardar_figura("e_sesgos_por_grupo"); plt.show()
tabla_sesgos

# %% [markdown]
# 🔎 **Interpretación:** El gráfico y la tabla muestran, para seis variables sensibles, cuántos estudiantes hay en cada grupo y su tasa de alto desempeño frente a la tasa global (línea punteada).
# En general, los datos contienen **inequidades de base** marcadas:
# - **Estrato:** la tasa pasa de 7,9 % (estrato 1) a 46,9 % (estrato 5); "Sin estrato" solo llega al 4,5 %.
# - **Naturaleza y subregión:** 31,5 % en colegios no oficiales frente a 13,5 % en oficiales; 21,2 % en el Valle de Aburrá frente a 11,3 % en el resto de Antioquia.
# - **Zona y género:** 18,7 % en zona urbana frente a 9,6 % en rural; 20,4 % en hombres frente a 14,5 % en mujeres.
# - **Jornada:** en la nocturna y la sabatina (jóvenes y adultos) la tasa es de apenas 2,1–2,2 %.
# - **Representación:** los colegios rurales (17,3 %), la jornada nocturna (4,9 %) y los estratos 5 y 6 (3,3 % entre ambos) son minoría.
#
# Estas diferencias son un **sesgo histórico**: reflejan desigualdades reales del sistema educativo, y el modelo las aprenderá. Por eso en 5.4 se audita su desempeño por grupo.
