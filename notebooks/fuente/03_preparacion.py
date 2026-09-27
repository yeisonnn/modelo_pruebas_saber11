# %% [markdown]
# ## 3. Preparación de los datos
# ### 3.1 Selección de variables

# %% [markdown]
# #### 📘 Concepto: Selección de variables
# Seleccionar variables reduce ruido, costo y riesgo de fuga. Hay tres familias de métodos:
# - **Por conocimiento del dominio:** se eligen variables con relación plausible con el objetivo y **disponibles al predecir**. Es el método principal aquí.
# - **De filtro:** se descartan variables constantes, identificadores y redundantes, o se ordenan por una medida estadística (χ², información mutua; ver 3.6).
# - **Envolventes o embebidos:** el propio modelo elige (p. ej. regularización L1, importancia en árboles).
#
# Además se **construyen** variables (*feature engineering*): `edad` a partir de la fecha de nacimiento y `cole_valle_aburra`
# a partir del código de municipio. Esto último reduce una variable de ~125 categorías (alta cardinalidad) a un indicador binario.

# %%
# 3.1 Conjunto de trabajo: variables seleccionadas + las dos derivadas (edad y Valle de Aburrá)
# Las 17 variables originales que entran al modelo (las otras 2 de FEATURES se construyen aquí)
VARS_ORIGINALES_SELECCIONADAS = [v for v in FEATURES if v not in ("edad", "cole_valle_aburra")]

# Se conservan también el estado del resultado (para filtrar) y el puntaje global (para construir el objetivo)
df_sel = df_raw[VARS_ORIGINALES_SELECCIONADAS + ["estu_estadoinvestigacion", "punt_global"]].copy()
df_sel["edad"] = calcular_edad(df_raw["estu_fechanacimiento"])                  # Derivada: edad en años cumplidos
df_sel["cole_valle_aburra"] = np.where(                                         # Derivada: área metropolitana
    df_raw["cole_cod_mcpio_ubicacion"].isin(CODIGOS_VALLE_ABURRA), "Sí", "No")
df_sel = df_sel[FEATURES + ["estu_estadoinvestigacion", "punt_global"]]         # Orden de columnas estable

# Justificación de dominio de cada variable seleccionada (tabla 3.1 del informe)
JUSTIFICACION_SELECCION = {
    "edad": "Derivada de la fecha de nacimiento; la extraedad se asocia con rezago escolar",
    "estu_genero": "Brechas de género documentadas en pruebas estandarizadas; permite auditar sesgo",
    "fami_estratovivienda": "Proxy del nivel socioeconómico del hogar",
    "fami_educacionmadre": "Capital cultural del hogar; predictor clásico del logro académico",
    "fami_educacionpadre": "Capital cultural del hogar",
    "fami_personashogar": "Tamaño del hogar (recursos per cápita)",
    "fami_cuartoshogar": "Condiciones de la vivienda (hacinamiento)",
    "fami_tieneinternet": "Acceso a recursos digitales de estudio",
    "fami_tienecomputador": "Acceso a recursos digitales de estudio",
    "fami_tieneautomovil": "Proxy de ingreso del hogar",
    "fami_tienelavadora": "Proxy de condiciones materiales del hogar",
    "cole_naturaleza": "Oficial vs. privado: diferencias de recursos y de selección de estudiantes",
    "cole_jornada": "Intensidad horaria y tipo de población (sabatina/noche = jóvenes y adultos)",
    "cole_area_ubicacion": "Brecha urbano-rural",
    "cole_bilingue": "Intensidad en inglés y recursos del colegio",
    "cole_caracter": "Orientación académica vs. técnica",
    "cole_genero": "Composición del colegio (mixto, femenino, masculino)",
    "cole_sede_principal": "Las sedes principales suelen concentrar más recursos",
    "cole_valle_aburra": "Resume 125 municipios en un indicador área metropolitana vs. resto de Antioquia",
}
tabla_seleccion = guardar_tabla(pd.DataFrame({"variable": FEATURES,
                                              "justificacion": [JUSTIFICACION_SELECCION[v] for v in FEATURES]}),
                                "seleccion_variables")
print(f"Variables de entrada: {len(FEATURES)} (numéricas: {len(VARIABLES_NUMERICAS)}, "
      f"ordinales: {len(VARIABLES_ORDINALES)}, nominales: {len(VARIABLES_NOMINALES)})")
tabla_seleccion

# %% [markdown]
# 🔎 **Interpretación:** Quedan 19 variables de entrada: 1 numérica (`edad`), 5 ordinales del hogar y 13 nominales del estudiante, el hogar y el colegio. Cada una tiene su justificación de dominio.
# Las dos variables **construidas** muestran el valor del *feature engineering*:
# - `edad` convierte una fecha de texto en una cantidad con sentido educativo (extraedad = rezago escolar).
# - `cole_valle_aburra` resume 125 municipios (alta cardinalidad) en un indicador binario interpretable.

# %% [markdown]
# ### 3.2 Descripción estadística y reporte exploratorio (ydata-profiling)

# %%
# 3.2.1 Descripción estadística del conjunto seleccionado (antes de limpiar, para evidenciar problemas)
df_perfil = df_sel[FEATURES + ["punt_global"]]   # Variables de entrada + el puntaje del que sale el objetivo
# Tabla 3.2 del informe: tamaño del conjunto y tipos de variables
tabla_descripcion = guardar_tabla(pd.DataFrame({
    "metrica": ["Número de registros", "Número de variables", "Variables numéricas", "Variables categóricas"],
    "resultado": [f"{len(df_perfil):,}", f"{df_perfil.shape[1]} ({len(FEATURES)} de entrada + punt_global)",
                  "2 (edad, punt_global)",
                  f"{len(VARIABLES_ORDINALES) + len(VARIABLES_NOMINALES)} "
                  f"({len(VARIABLES_ORDINALES)} ordinales + {len(VARIABLES_NOMINALES)} nominales)"]}),
    "descripcion_estadistica")
display(tabla_descripcion)
# include="all": numéricas (media, desviación, cuartiles) y categóricas (únicos, moda y su frecuencia)
df_perfil.describe(include="all").T

# %% [markdown]
# 🔎 **Interpretación:** El conjunto seleccionado tiene 144.766 registros y 20 variables (19 de entrada más el puntaje global): 2 numéricas y 18 categóricas.
# La descripción confirma lo visto en la fase 2: las categorías más frecuentes son estrato 2, madre con bachillerato completo, colegio oficial y jornada de la mañana. La edad tiene media 17,5 años, pero un **mínimo de 0**, evidencia directa de fechas erróneas. Estos datos todavía no se han limpiado a propósito: el perfilamiento debe documentar el estado original.

# %% [markdown]
# #### 📘 Concepto: Perfilamiento automático
# Un **perfilamiento** (*data profiling*) resume de forma sistemática cada variable. **ydata-profiling** genera un reporte
# HTML con:
# - Tipo inferido, % de faltantes, cardinalidad, valores más frecuentes e histogramas.
# - **Alertas**: alta correlación, desbalance, valores constantes, duplicados y ceros.
# - Matrices de correlación (Pearson, Spearman, Cramér, φK) y diagramas de faltantes.
#
# No reemplaza el análisis del científico de datos, pero lo acelera y deja evidencia auditable. Se ejecuta **antes de
# limpiar** para documentar el estado original de los datos.

# %%
# 3.2.2 Reporte exploratorio automático con ydata-profiling (se adjunta como HTML al informe)
import os
os.environ["TQDM_DISABLE"] = "1"   # Oculta las barras de progreso internas de la librería (no aportan al análisis)

inicio = time.time()
with warnings.catch_warnings():    # Silencia solo aquí los avisos internos de la librería (deprecación, copias de pandas)
    warnings.simplefilter("ignore")
    from ydata_profiling import ProfileReport   # Import local: la librería es pesada y solo se usa aquí
    # interactions desactivado: los diagramas de dispersión por pares cuestan O(p²) y aportan poco con variables categóricas
    perfil = ProfileReport(df_perfil, title="Saber 11 · Antioquia 2022-2 · Variables seleccionadas",
                           interactions={"continuous": False}, progress_bar=False)
    RUTA_PERFIL = DIR_REPORTES / "perfilamiento_saber11_antioquia.html"
    perfil.to_file(RUTA_PERFIL)   # Entregable exigido por la actividad
print(f"Reporte: {RUTA_PERFIL.name} ({RUTA_PERFIL.stat().st_size / 1e6:.1f} MB) en {time.time() - inicio:.0f} s")

# %% [markdown]
# 🔎 **Interpretación:** Se generó `reports/perfilamiento_saber11_antioquia.html` (1,8 MB, en menos de un minuto). Contiene, por variable, histogramas, faltantes, cardinalidad y alertas, además de las matrices de correlación.
# Al importarla, la librería emite un aviso de deprecación (silenciado en la celda): ydata-profiling dejará de actualizarse en favor de `fg-data-profiling`. Se mantiene porque la actividad lo exige explícitamente, y es una buena lección sobre el ciclo de vida de las librerías de código abierto. **Recomendación de estudio:** abre el HTML y compara sus alertas de desbalance y faltantes con lo que encontramos manualmente en la fase 2.

# %% [markdown]
# ### 3.3 Limpieza de errores

# %% [markdown]
# #### 📘 Concepto: Limpieza determinística
# Hay dos clases de transformaciones y el **momento** en que se aplican importa:
#
# | Tipo | Ejemplo | ¿Aprende de los datos? | Cuándo aplicarla |
# |---|---|---|---|
# | **Determinística** | Eliminar duplicados, corregir tildes, marcar edades imposibles | No: la regla es fija | Antes de partir los datos |
# | **Aprendida** | Imputar con la mediana, escalar, codificar, SMOTE | Sí: calcula estadísticos | **Dentro del pipeline**, ajustada solo con entrenamiento |
#
# Si una transformación aprendida se ajusta con todos los datos, la información del conjunto de prueba "se filtra" al
# entrenamiento y las métricas quedan optimistas. Las reglas determinísticas no tienen ese problema: dan el mismo resultado
# para un registro sin importar los demás.

# %%
# 3.3 Reglas determinísticas (no aprenden de los datos), por eso se aplican antes de la partición
# Variantes de escritura conocidas -> valor canónico (se amplía si la tabla 2.2.2 muestra otras)
HOMOLOGACIONES = {"fami_cuartoshogar": {"Seis o más": "Seis o mas"}, "fami_personashogar": {"9 o mas": "9 o más"}}
df_limpio = df_sel.copy()   # Se trabaja sobre una copia para conservar df_sel como evidencia del estado original
registro = []               # Bitácora de cada regla: (error, acción, registros afectados) -> tabla 3.3 del informe

# (a) Duplicados exactos detectados con las 51 columnas originales (dos estudiantes pueden compartir el mismo perfil)
mascara = df_raw.loc[df_limpio.index].duplicated(keep="first")
df_limpio = df_limpio.loc[~mascara]
registro.append(("Filas duplicadas exactas (51 columnas)", "Eliminadas; se conserva la primera ocurrencia", int(mascara.sum())))

# (b) Resultados no oficiales: el ICFES los marca con un estado distinto de PUBLICAR
mascara = df_limpio["estu_estadoinvestigacion"] != "PUBLICAR"
df_limpio = df_limpio.loc[~mascara]
registro.append(("Resultado en investigación", "Eliminados: el puntaje no es oficial", int(mascara.sum())))

# (c) Puntaje global ausente o fuera de [0, 500]: sin él no hay variable objetivo confiable
mascara = df_limpio["punt_global"].isna() | ~df_limpio["punt_global"].between(0, 500)
df_limpio = df_limpio.loc[~mascara]
registro.append(("Puntaje global nulo o fuera de [0, 500]", "Eliminados: no se puede construir el objetivo", int(mascara.sum())))

# (d) Edad imposible: no se elimina el registro (el resto de su información es válida); se marca como faltante
mascara = df_limpio["edad"].notna() & ~df_limpio["edad"].between(13, 30)
df_limpio.loc[mascara, "edad"] = np.nan
registro.append(("Edad fuera de [13, 30]", "Convertida a NaN (imputación por mediana en el pipeline)", int(mascara.sum())))

# (e) Texto: espacios sobrantes y variantes de escritura; (f) valores fuera del dominio válido -> NaN
total_invalidas = 0
for var, validas in CATEGORIAS_VALIDAS.items():
    df_limpio[var] = df_limpio[var].str.strip().replace(HOMOLOGACIONES.get(var, {}))   # Normaliza antes de validar
    mascara = df_limpio[var].notna() & ~df_limpio[var].isin(validas)                   # Valores que siguen siendo inválidos
    total_invalidas += int(mascara.sum())
    df_limpio.loc[mascara, var] = np.nan
registro.append(("Categorías fuera del dominio válido", "Homologadas si son variantes; si no, NaN", total_invalidas))

# Construcción de la variable objetivo (regla de negocio de la fase 1)
df_limpio[OBJETIVO] = (df_limpio["punt_global"] >= UMBRAL_ALTO_DESEMPENO).astype(int)
tabla_errores = guardar_tabla(pd.DataFrame(registro, columns=["tipo_error", "accion", "registros_afectados"]),
                              "limpieza_errores")
print(f"Registros: {len(df_sel):,} -> {len(df_limpio):,}")
tabla_errores

# %% [markdown]
# 🔎 **Interpretación:** La limpieza pasa de 144.766 a **72.357 registros**:
# - Se eliminan 72.383 duplicados exactos y 26 resultados no oficiales.
# - Ningún puntaje estaba nulo ni fuera de rango.
# - 1.126 edades imposibles pasan a faltante (la mitad de las 2.252 detectadas, por los duplicados).
# - Ninguna categoría estaba fuera de dominio.
#
# Es el caso típico de **limpieza determinística** (📘): cada regla da el mismo resultado para un registro sin mirar los demás, así que se puede aplicar antes de partir los datos sin riesgo de fuga. La tabla es la 3.3 del informe.

# %%
# 3.3.1 Verificación automática: las reglas de calidad se cumplen después de limpiar
# Los assert detienen el cuaderno si alguna regla falla: la limpieza se prueba, no se supone
assert not df_raw.loc[df_limpio.index].duplicated().any(), "Persisten duplicados"
assert (df_limpio["estu_estadoinvestigacion"] == "PUBLICAR").all()     # Solo resultados oficiales
assert df_limpio["punt_global"].between(0, 500).all()                   # Objetivo válido en todas las filas
assert df_limpio["edad"].dropna().between(13, 30).all()                 # Edades plausibles (los NaN se imputarán)
for var, validas in CATEGORIAS_VALIDAS.items():
    assert df_limpio[var].dropna().isin(validas).all(), var            # Dominios categóricos respetados
print("Todas las reglas de calidad se cumplen en df_limpio")

# %% [markdown]
# 🔎 **Interpretación:** Todas las aserciones pasan: no quedan duplicados, solo hay resultados oficiales, el objetivo es válido en todas las filas, las edades son plausibles y las categorías están dentro de su dominio.
# Convertir las reglas de calidad en `assert` es una práctica de **pruebas de datos**. Si la fuente cambiara y reapareciera un problema, el cuaderno se detendría aquí en lugar de entrenar un modelo con datos defectuosos.

# %% [markdown]
# ### 3.4 Limpieza de valores nulos

# %% [markdown]
# #### 📘 Concepto: Valores faltantes
# Rubin (1976) clasifica los faltantes según su **mecanismo**:
# - **MCAR** (completamente al azar): la ausencia no depende de nada. Eliminar filas no sesga, pero pierde información.
# - **MAR** (al azar condicionado): la ausencia depende de otras variables observadas. Se puede imputar usándolas.
# - **MNAR** (no al azar): la ausencia depende del propio valor faltante. Por ejemplo, quien no reporta el estrato puede tener
#   un perfil socioeconómico particular.
#
# Estrategias usadas aquí:
# - **Imputación por mediana:** robusta a valores extremos (numéricas y ordinales).
# - **Indicador de faltante:** columna 0/1 que permite al modelo aprender si "no tener dato" es informativo (útil en MNAR).
# - **Categoría explícita `DESCONOCIDO`:** para nominales; conserva la señal del faltante sin inventar un valor.
#
# Toda imputación **aprende** un estadístico, así que va dentro del pipeline (ver 3.3).

# %%
# 3.4 Respuestas no informativas -> NaN y diagnóstico del % de faltantes (la imputación ocurre en el pipeline)
# "No sabe" / "No Aplica" no son un nivel educativo: no tienen lugar en la escala ordinal, así que son faltantes
for var in ["fami_educacionmadre", "fami_educacionpadre"]:
    df_limpio[var] = df_limpio[var].replace(["No sabe", "No Aplica"], np.nan)

# Método de imputación por tipo de variable (tabla 3.4 del informe); los ** fusionan los tres diccionarios
METODO_IMPUTACION = {**{v: "Mediana (SimpleImputer dentro del pipeline)" for v in VARIABLES_NUMERICAS},
                     **{v: "Mediana del código ordinal + indicador de faltante (pipeline)" for v in VARIABLES_ORDINALES},
                     **{v: "Categoría explícita 'DESCONOCIDO' (pipeline)" for v in VARIABLES_NOMINALES}}
pct_nulos = (df_limpio[FEATURES].isna().mean() * 100).round(2)   # isna().mean() = proporción de faltantes
tabla_nulos = guardar_tabla(pd.DataFrame({"variable": FEATURES, "pct_nulos": pct_nulos.values,
                                          "metodo": [METODO_IMPUTACION[v] for v in FEATURES]})
                            .sort_values("pct_nulos", ascending=False), "limpieza_nulos")

fig, eje = plt.subplots(figsize=(9, 6))
pct_nulos.sort_values().plot.barh(ax=eje, color="#4C72B0")   # Barras horizontales ordenadas: fácil de comparar
eje.set_xlabel("% de valores nulos"); eje.set_title("Nulos por variable (después de limpiar errores)")
plt.tight_layout(); guardar_figura("nulos_por_variable"); plt.show()
tabla_nulos

# %% [markdown]
# 🔎 **Interpretación:** Tras convertir "No sabe" y "No Aplica" en faltantes, los mayores porcentajes de nulos son:
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
# Ninguna variable supera el 20 %, así que se **imputan** en lugar de descartarse. La imputación ocurre dentro del pipeline, como indica el 📘.

# %%
# 3.4.1 ¿El faltante es informativo? Tasa de alto desempeño con y sin dato (justifica DESCONOCIDO e indicadores)
# Si las tasas difieren mucho, el faltante NO es MCAR: eliminar esas filas sesgaría la muestra
filas = []
for var in pct_nulos[pct_nulos > 1].index:   # Solo variables con más de 1 % de faltantes
    falta = df_limpio[var].isna()
    filas.append({"variable": var, "pct_nulos": pct_nulos[var],
                  "tasa_alto_con_dato_pct": 100 * df_limpio.loc[~falta, OBJETIVO].mean(),
                  "tasa_alto_sin_dato_pct": 100 * df_limpio.loc[falta, OBJETIVO].mean()})
pd.DataFrame(filas).round(2)

# %% [markdown]
# 🔎 **Interpretación:** La prueba es contundente:
# - Quienes **no tienen dato de edad** alcanzan alto desempeño solo en el 0,8 % de los casos, frente al 17,4 %.
# - Sin estrato reportado, la tasa es 10,3 % frente a 17,6 %; sin dato de internet, 9,9 % frente a 17,6 %.
#
# Los faltantes **no son MCAR**: estar incompleto se asocia a un perfil de menor desempeño, un caso de MNAR/MAR (📘). Eliminar esas filas sesgaría la muestra hacia estudiantes más favorecidos. La categoría `DESCONOCIDO` y los indicadores de faltante permiten que el modelo aproveche esta señal.

# %%
# 3.4.2 Guardado del conjunto limpio (insumo reproducible para las fases siguientes)
# index_label="id_registro" conserva la posición original en df_raw: permite rastrear cualquier fila hasta la fuente
df_limpio.to_csv(DIR_PROC / "saber11_antioquia_limpio.csv.gz", index=True, index_label="id_registro", compression="gzip")
print(df_limpio.shape, "| tasa de alto desempeño:", f"{df_limpio[OBJETIVO].mean():.2%}")

# %% [markdown]
# 🔎 **Interpretación:** Se guardó el conjunto limpio (72.357 registros × 22 columnas), con una tasa de alto desempeño del 17,13 %: prácticamente la misma que antes de limpiar. La columna `id_registro` conserva la posición original, de modo que cualquier fila puede rastrearse hasta la descarga de la API (**trazabilidad**).

# %% [markdown]
# ### 3.5 Codificación de variables categóricas

# %% [markdown]
# #### 📘 Concepto: Partición entrenamiento/prueba
# Para estimar cómo funcionará el modelo con datos **que nunca vio**, se reserva un **conjunto de prueba** (aquí el 30 %)
# que no se toca hasta la fase 5. El **entrenamiento** (70 %) se usa para ajustar el preprocesamiento, comparar modelos
# con validación cruzada y ajustar hiperparámetros.
# - **Estratificar** (`stratify=y`) conserva la proporción de clases (~17/83) en ambos conjuntos. Sin estratificar, un
#   conjunto podría quedar con menos positivos por azar.
# - **Muestra de modelado:** por costo computacional se modela con 30.000 registros estratificados. El SVM con kernel RBF
#   escala entre O(n²) y O(n³). El resto (**pool**) sirve después como datos nunca vistos para validar el despliegue.

# %%
# 3.5.1 Muestra de modelado estratificada (30.000) y partición 70/30 estratificada
# train_size con un entero = número exacto de registros; df_pool recibe el resto (nunca participa en el modelado)
df_modelo, df_pool = train_test_split(df_limpio, train_size=N_MUESTRA_MODELADO,
                                      stratify=df_limpio[OBJETIVO], random_state=RANDOM_STATE)
X, y = df_modelo[FEATURES], df_modelo[OBJETIVO]   # X = matriz de entrada; y = vector objetivo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

# Verificación: la tasa de positivos debe ser prácticamente igual en todos los conjuntos (efecto de estratificar)
pd.DataFrame({"conjunto": ["Limpio total", "Muestra de modelado", "Entrenamiento (70 %)", "Prueba (30 %)", "Pool no utilizado"],
              "registros": [len(df_limpio), len(df_modelo), len(X_train), len(X_test), len(df_pool)],
              "pct_alto_desempeno": [100 * s.mean() for s in [df_limpio[OBJETIVO], y, y_train, y_test, df_pool[OBJETIVO]]]}).round(2)

# %% [markdown]
# 🔎 **Interpretación:** La estratificación funciona: la tasa de positivos es 17,13 % en todos los conjuntos. Los tamaños quedan así:
#
# | Conjunto | Registros |
# |---|---|
# | Muestra de modelado | 30.000 |
# | Entrenamiento | 21.000 |
# | Prueba | 9.000 |
# | Pool que no participa en el modelado | 42.357 |
#
# El conjunto de prueba queda "guardado bajo llave" hasta la fase 5, y el pool servirá en la fase 6 como datos reales nunca vistos.

# %% [markdown]
# #### 📘 Concepto: Codificación y escalado
# Los algoritmos operan con números, así que las categorías deben **codificarse**:
#
# | Técnica | Cómo funciona | Cuándo usarla |
# |---|---|---|
# | **Ordinal** | Asigna 0, 1, 2… respetando un orden explícito | Categorías con orden natural (estrato, nivel educativo) |
# | **One-Hot** | Crea una columna 0/1 por categoría | Categorías **sin** orden (jornada, naturaleza). Codificarlas como 0, 1, 2 inventaría distancias falsas |
#
# El **escalado** (estandarización) $z = \frac{x-\mu}{\sigma}$ pone las variables en la misma escala. Es indispensable
# para los modelos basados en distancias o márgenes (KNN, SVM) y para la regresión logística regularizada. A los árboles no
# les afecta.
#
# `ColumnTransformer` aplica a cada grupo de columnas su propio `Pipeline` (imputar → codificar → escalar). Al ir dentro del
# pipeline del modelo, cada estadístico ($\mu$, $\sigma$, mediana, categorías) se aprende **solo con entrenamiento**.

# %%
# 3.5.2 Preprocesamiento por tipo de variable (ColumnTransformer con transformadores nativos de scikit-learn)
# Orden explícito de cada variable ordinal: el primer elemento recibe el código 0, el siguiente 1, etc.
ORDEN_ORDINALES = {
    "fami_estratovivienda": ["Sin Estrato", "Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"],
    "fami_educacionmadre": EDUCACION_ORDEN,
    "fami_educacionpadre": EDUCACION_ORDEN,
    "fami_personashogar": ["1 a 2", "3 a 4", "5 a 6", "7 a 8", "9 o más"],
    "fami_cuartoshogar": ["Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis o mas"],
}
# Numéricas: mediana (robusta a extremos) y luego estandarización z-score
pipe_numerica = Pipeline([("imputar", SimpleImputer(strategy="median")), ("escalar", StandardScaler())])
pipe_ordinal = Pipeline([
    # Categoría no vista o faltante -> NaN, que el paso siguiente imputa (así "No sabe" no rompe la escala)
    ("codificar", OrdinalEncoder(categories=[ORDEN_ORDINALES[v] for v in VARIABLES_ORDINALES],
                                 handle_unknown="use_encoded_value", unknown_value=np.nan)),
    ("imputar", SimpleImputer(strategy="median", add_indicator=True)),   # add_indicator agrega la columna "faltaba"
    ("escalar", StandardScaler())])
pipe_nominal = Pipeline([
    ("imputar", SimpleImputer(strategy="constant", fill_value="DESCONOCIDO")),   # El faltante se vuelve una categoría
    # handle_unknown="ignore": una categoría nueva en producción se codifica como todo ceros, sin error
    ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
# Cada tupla: (nombre, transformación, columnas); remainder="drop" descarta cualquier otra columna
preprocesador = ColumnTransformer([("num", pipe_numerica, VARIABLES_NUMERICAS),
                                   ("ord", pipe_ordinal, VARIABLES_ORDINALES),
                                   ("nom", pipe_nominal, VARIABLES_NOMINALES)], remainder="drop")

# Método de codificación por variable (tabla 3.5 del informe)
METODO_CODIFICACION = {
    **{v: "Numérica: imputación por mediana + estandarización (z-score)" for v in VARIABLES_NUMERICAS},
    **{v: f"Ordinal con orden explícito ({ORDEN_ORDINALES[v][0]} < … < {ORDEN_ORDINALES[v][-1]}) + indicador de faltante + estandarización"
       for v in VARIABLES_ORDINALES},
    **{v: "One-Hot Encoding (una columna por categoría, incluida DESCONOCIDO)" for v in VARIABLES_NOMINALES}}
tabla_codificacion = guardar_tabla(pd.DataFrame({"variable": FEATURES, "metodo": [METODO_CODIFICACION[v] for v in FEATURES]}),
                                   "codificacion")
tabla_codificacion

# %% [markdown]
# 🔎 **Interpretación:** La tabla resume la estrategia de codificación por tipo de variable (tabla 3.5 del informe):
# - **Edad:** mediana y estandarización.
# - **5 ordinales:** código ordinal con orden explícito (p. ej. "Sin Estrato" < "Estrato 1" < … < "Estrato 6"), indicador de faltante y estandarización.
# - **13 nominales:** One-Hot con la categoría `DESCONOCIDO`.
#
# El `ColumnTransformer` no se ha ajustado todavía: es una "receta" que cada pipeline aprenderá **solo con sus datos de entrenamiento**.

# %%
# 3.5.3 Ajuste del preprocesamiento SOLO con entrenamiento y vista de la matriz codificada
# clone crea una copia sin entrenar: el objeto "preprocesador" original queda limpio para usarlo dentro de los pipelines
preprocesador_demo = clone(preprocesador).fit(X_train)
X_train_cod = pd.DataFrame(preprocesador_demo.transform(X_train),
                           columns=preprocesador_demo.get_feature_names_out(),   # Nombres tipo "nom__cole_jornada_MAÑANA"
                           index=X_train.index)
print(f"Variables originales: {X_train.shape[1]} -> variables codificadas: {X_train_cod.shape[1]}")
X_train_cod.head()   # Numéricas y ordinales estandarizadas; nominales como columnas 0/1

# %% [markdown]
# 🔎 **Interpretación:** Ajustado con el entrenamiento, el preprocesamiento convierte 19 variables en **50 columnas numéricas**: 1 de edad, 5 ordinales con sus 5 indicadores de faltante y 39 columnas one-hot.
# En la vista se observa que las ordinales y los indicadores quedan estandarizados (media 0), mientras las nominales quedan como 0/1. Los nombres (`ord__…`, `nom__cole_jornada_MAÑANA`) permiten rastrear cada columna hasta su variable original.

# %% [markdown]
# ### 3.6 Relaciones lineales y no lineales

# %% [markdown]
# #### 📘 Concepto: Correlación de Pearson y Spearman
# - **Pearson** mide la relación **lineal** entre dos variables numéricas:
#   $$r = \frac{\sum_i (x_i-\bar x)(y_i-\bar y)}{\sqrt{\sum_i (x_i-\bar x)^2}\,\sqrt{\sum_i (y_i-\bar y)^2}} \in [-1, 1]$$
# - **Spearman** ($\rho$) es la correlación de Pearson calculada sobre los **rangos**. Mide relaciones **monótonas**
#   (siempre crecientes o siempre decrecientes), aunque no sean lineales, y es robusta a valores extremos. Es la adecuada
#   para variables **ordinales** como el estrato.
#
# Si $|\rho|$ es claramente mayor que $|r|$, la relación es monótona pero no lineal. Correlación **no implica causalidad**:
# el estrato y el puntaje pueden estar relacionados porque ambos dependen del ingreso del hogar.

# %%
# 3.6.1 Correlación lineal (Pearson) y monótona (Spearman) entre variables numéricas/ordinales y el puntaje
df_analisis = X_train.copy()   # Solo entrenamiento: el análisis exploratorio tampoco debe "ver" el conjunto de prueba
for var, orden in ORDEN_ORDINALES.items():
    df_analisis[var] = df_analisis[var].map({cat: i for i, cat in enumerate(orden)})   # Código ordinal para analizar
df_analisis["punt_global"] = df_modelo.loc[X_train.index, "punt_global"]   # Puntaje continuo (más información que 0/1)
cols_corr = VARIABLES_NUMERICAS + VARIABLES_ORDINALES + ["punt_global"]

fig, ejes = plt.subplots(1, 2, figsize=(16, 6))
for eje, metodo in zip(ejes, ["pearson", "spearman"]):   # Mismo mapa de calor con cada método, para comparar
    sns.heatmap(df_analisis[cols_corr].corr(method=metodo), annot=True, fmt=".2f", cmap="RdBu_r",
                vmin=-1, vmax=1, ax=eje)                  # Escala fija [-1, 1]: rojo positivo, azul negativo
    eje.set_title(f"Correlación de {metodo.capitalize()}")
plt.tight_layout(); guardar_figura("correlaciones"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** Hallazgos:
# - **Educación de los padres:** es la asociación más fuerte con el puntaje (madre r = 0,39 y ρ = 0,37; padre r = 0,37 y ρ = 0,35).
# - **Edad:** relación negativa (r = −0,28, ρ = −0,30): a mayor edad, menor puntaje.
# - **Estrato:** asociación positiva moderada (0,26).
# - **Multicolinealidad:** las dos educaciones están muy correlacionadas entre sí (0,60), y el número de personas y de cuartos del hogar también (0,47).
# - **Forma de la relación:** Pearson y Spearman son muy parecidos, así que las relaciones son **aproximadamente monótonas**. Ninguna variable ordinal muestra una curvatura que Pearson esté subestimando.
#
# Como advierte el 📘, son asociaciones, no causas. El estrato y la educación de los padres comparten un factor común: el nivel socioeconómico del hogar.

# %% [markdown]
# #### 📘 Concepto: V de Cramér e información mutua
# - **V de Cramér** mide la asociación entre dos variables **categóricas** a partir del estadístico $\chi^2$ de la tabla de contingencia:
#   $$V = \sqrt{\frac{\chi^2}{n\,(\min(r,k)-1)}} \in [0,1]$$
#   donde $r$ y $k$ son el número de filas y columnas. El p-valor de $\chi^2$ indica si la asociación es estadísticamente
#   distinta de cero. Con $n$ grande casi todo es "significativo", por eso importa el **tamaño** $V$.
# - **Información mutua** mide cuánta incertidumbre sobre $Y$ se reduce al conocer $X$:
#   $$I(X;Y) = \sum_{x,y} p(x,y)\,\log\frac{p(x,y)}{p(x)\,p(y)} \ge 0$$
#   Es cero solo si las variables son independientes y capta **cualquier** tipo de dependencia, lineal o no.
#   Por eso complementa a Pearson.

# %%
# 3.6.2 Asociación con el objetivo: V de Cramér (χ²) e información mutua (captura relaciones no lineales)
def v_cramer(x, y):
    """V de Cramér y p-valor de χ² entre dos variables categóricas."""
    tabla = pd.crosstab(x, y)                                   # Tabla de contingencia (frecuencias conjuntas)
    chi2, p_valor, _, _ = stats.chi2_contingency(tabla)         # Prueba de independencia χ²
    n = tabla.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(tabla.shape) - 1))), p_valor


# mutual_info_classif necesita números: se codifica cada categoría con un entero (el orden no importa en la IM discreta)
X_mi = pd.DataFrame(index=X_train.index)
for var in FEATURES:
    if var in VARIABLES_NUMERICAS:
        X_mi[var] = X_train[var].fillna(X_train[var].median())            # Continua: imputación simple solo para el análisis
    else:
        X_mi[var] = pd.factorize(X_train[var].fillna("DESCONOCIDO"))[0]   # Discreta: entero por categoría
info_mutua = mutual_info_classif(X_mi[FEATURES], y_train, random_state=RANDOM_STATE,
                                 discrete_features=[v not in VARIABLES_NUMERICAS for v in FEATURES])
filas = []
for var, mi in zip(FEATURES, info_mutua):
    v, p_valor = v_cramer(X_train[var].fillna("DESCONOCIDO").astype(str), y_train)
    # Spearman con el puntaje solo tiene sentido para numéricas y ordinales
    rho = df_analisis[var].corr(df_analisis["punt_global"], method="spearman") if var in cols_corr else np.nan
    filas.append({"variable": var, "v_cramer": v, "p_chi2": p_valor, "informacion_mutua": mi, "spearman_punt_global": rho})
tabla_relaciones = guardar_tabla(pd.DataFrame(filas).sort_values("informacion_mutua", ascending=False).round(4), "relaciones")

fig, ejes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)   # Mismo eje Y: se comparan ambos rankings por variable
orden_vars = tabla_relaciones["variable"]
ejes[0].barh(orden_vars, tabla_relaciones["informacion_mutua"], color="#4C72B0"); ejes[0].set_title("Información mutua con el objetivo")
ejes[1].barh(orden_vars, tabla_relaciones["v_cramer"], color="#DD8452"); ejes[1].set_title("V de Cramér con el objetivo")
ejes[0].invert_yaxis(); plt.tight_layout(); guardar_figura("asociacion_objetivo"); plt.show()   # La más asociada arriba
tabla_relaciones

# %% [markdown]
# 🔎 **Interpretación:** Todas las variables tienen asociación estadísticamente significativa con el objetivo (p < 0,001). Con n = 21.000 esto es esperable, así que lo relevante es el **tamaño** del efecto:
# - Las más asociadas son la educación de la madre (V = 0,30; IM = 0,043) y del padre (V = 0,29), la edad, tener computador, el estrato y la jornada.
# - Las menos asociadas son la sede principal, el género del estudiante y el número de cuartos (V < 0,08).
#
# La información mutua y la V de Cramér ordenan las variables de forma muy parecida. La información es moderada y está repartida entre varias variables, lo que anticipa un techo de desempeño modesto para cualquier modelo.

# %%
# 3.6.3 Evidencia de no linealidad: puntaje medio por edad y tasa de alto desempeño por estrato
fig, ejes = plt.subplots(1, 2, figsize=(15, 4.5))
# Izquierda: si la curva no es una recta, un coeficiente lineal subestima la relación edad-puntaje
por_edad = df_analisis.groupby("edad")["punt_global"].agg(["mean", "size"]).query("size >= 30")   # Grupos estables
ejes[0].plot(por_edad.index, por_edad["mean"], marker="o"); ejes[0].set_xlabel("Edad"); ejes[0].set_ylabel("Puntaje global medio")
ejes[0].set_title("Puntaje medio por edad (grupos con ≥ 30 registros)")
# Derecha: tasa de positivos por estrato, en el orden natural del estrato
por_estrato = pd.Series(y_train.groupby(X_train["fami_estratovivienda"].fillna("Sin dato")).mean() * 100)
por_estrato = por_estrato.reindex([c for c in ORDEN_ORDINALES["fami_estratovivienda"] + ["Sin dato"] if c in por_estrato.index])
ejes[1].bar(por_estrato.index, por_estrato.values, color="#4C72B0"); ejes[1].set_ylabel("% alto desempeño")
ejes[1].set_title("Tasa de alto desempeño por estrato"); ejes[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); guardar_figura("no_linealidad"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** - **Edad (izquierda):** la relación **no es lineal**. El puntaje medio cae con fuerza entre los 15 y los 20 años (de ~263 a ~210) y luego se estabiliza alrededor de 200–210. Un coeficiente lineal único no describe bien esa forma: es la no linealidad que la información mutua sí capta.
# - **Estrato (derecha):** crece de forma monótona hasta el estrato 5 (~46 %) y baja ligeramente en el estrato 6 (~42 %, con pocos casos). "Sin estrato" es el grupo más bajo (~5 %).
#
# Por eso tiene sentido comparar modelos lineales con modelos flexibles (árboles, SVM RBF) en la fase 4.

# %% [markdown]
# ### 3.7 Reducción de dimensiones (PCA)

# %% [markdown]
# #### 📘 Concepto: Análisis de Componentes Principales
# **PCA** busca nuevas variables (componentes) que son **combinaciones lineales** de las originales, no están correlacionadas
# entre sí y están ordenadas por la **varianza** que explican. Matemáticamente, son los vectores propios de la matriz de
# covarianza $\Sigma = \frac{1}{n}X^\top X$ (con $X$ centrada), y la fracción de varianza que explica la componente $k$ es
# $\lambda_k / \sum_j \lambda_j$.
#
# - **Ventajas:** reduce dimensión, ruido y colinealidad, y acelera modelos basados en distancias.
# - **Desventajas:** las componentes pierden interpretabilidad ("0,3·estrato − 0,2·jornada…"), PCA solo capta estructura
#   lineal y maximiza varianza, no poder predictivo.
#
# **Regla de decisión:** aplicarlo solo si mejora el desempeño o si la dimensión es un problema. Aquí se evalúa empíricamente.

# %%
# 3.7.1 Varianza explicada por componentes principales sobre la matriz codificada de entrenamiento
pca = PCA(random_state=RANDOM_STATE).fit(X_train_cod)      # Sin n_components: calcula todas las componentes
var_acum = np.cumsum(pca.explained_variance_ratio_)        # Varianza explicada acumulada
n_90 = int(np.searchsorted(var_acum, 0.90) + 1)            # Mínimo de componentes para llegar al 90 %
n_95 = int(np.searchsorted(var_acum, 0.95) + 1)            # ... y al 95 %

fig, eje = plt.subplots(figsize=(9, 4))
eje.plot(range(1, len(var_acum) + 1), var_acum, marker="o", ms=3)   # Curva de varianza acumulada
eje.axvline(n_90, ls="--", color="crimson", label=f"90 % con {n_90} componentes")
eje.axvline(n_95, ls=":", color="crimson", label=f"95 % con {n_95} componentes")
eje.set_xlabel("Número de componentes"); eje.set_ylabel("Varianza explicada acumulada")
eje.set_title("PCA sobre las variables codificadas"); eje.legend()
plt.tight_layout(); guardar_figura("pca_varianza"); plt.show()
print(f"Dimensión codificada: {X_train_cod.shape[1]} | 90 %: {n_90} componentes | 95 %: {n_95} componentes")

# %% [markdown]
# 🔎 **Interpretación:** Con 50 columnas codificadas, PCA necesita **17 componentes para explicar el 90 %** de la varianza y 21 para el 95 %. La curva crece de forma gradual, sin un "codo" marcado.
# La varianza está repartida porque muchas columnas son dummies casi independientes. Reducir a 17 componentes quitaría dimensiones, pero cada componente mezclaría variables heterogéneas (jornada, estrato, tenencias) y perdería su significado.

# %%
# 3.7.2 Prueba empírica: ¿PCA mejora el F1 de un modelo de referencia? (regresión logística balanceada, CV 5-fold)
# Se comparan dos pipelines idénticos salvo por el paso PCA; la decisión se basa en evidencia, no en costumbre
referencia = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
f1_sin_pca = cross_val_score(Pipeline([("prep", clone(preprocesador)), ("modelo", clone(referencia))]),
                             X_train, y_train, cv=CV, scoring="f1", n_jobs=-1)
f1_con_pca = cross_val_score(Pipeline([("prep", clone(preprocesador)),
                                       ("pca", PCA(n_components=0.90, random_state=RANDOM_STATE)),   # 0.90 = 90 % de varianza
                                       ("modelo", clone(referencia))]),
                             X_train, y_train, cv=CV, scoring="f1", n_jobs=-1)
APLICAR_PCA = bool(f1_con_pca.mean() > f1_sin_pca.mean() + 0.01)   # Solo si mejora el F1 en más de 0,01
justificacion_pca = (
    f"PCA: {n_90} de {X_train_cod.shape[1]} componentes explican el 90 % de la varianza. "
    f"F1 CV sin PCA = {f1_sin_pca.mean():.4f}; con PCA = {f1_con_pca.mean():.4f}. "
    + ("Se aplica porque mejora el F1 en más de 0,01." if APLICAR_PCA else
       "No se aplica: no mejora el desempeño, la dimensión ya es baja y se perdería la interpretabilidad de las variables."))
guardar_tabla(pd.DataFrame({"tecnica": ["Análisis de Componentes Principales (PCA)"], "justificacion": [justificacion_pca]}),
              "reduccion_dimensiones")   # Tabla 3.6 del informe
print(justificacion_pca)

# %% [markdown]
# 🔎 **Interpretación:** La prueba empírica decide: con PCA el F1 en validación cruzada **baja** de 0,4685 a 0,4592. Como no mejora (el criterio exigía +0,01), **no se aplica** reducción de dimensiones.
# Es la regla de decisión del 📘: PCA maximiza varianza, no poder predictivo, y aquí descarta parte de la señal útil para la clase minoritaria. Además, conservar las variables originales permite interpretar el modelo en la fase 5.

# %% [markdown]
# ### 3.8 Balanceo de clases

# %% [markdown]
# #### 📘 Concepto: Balanceo de clases
# Con clases desbalanceadas, el modelo "aprende" que casi siempre conviene predecir la mayoritaria. Estrategias:
#
# | Estrategia | Idea | Ventaja | Riesgo |
# |---|---|---|---|
# | **Ponderación** (`class_weight`) | Penaliza más los errores en la minoritaria | No altera los datos | No todos los modelos lo admiten (p. ej. KNN) |
# | **Submuestreo** (RandomUnderSampler) | Descarta registros de la mayoritaria | Rápido | Pierde información |
# | **SMOTE** (Chawla et al., 2002) | Crea registros sintéticos interpolando vecinos de la minoritaria | Conserva todos los datos | Puede generar ruido en zonas de solapamiento |
#
# SMOTE genera cada registro sintético como $x_{nuevo} = x_i + \lambda\,(x_{vecino} - x_i)$ con $\lambda \sim U(0,1)$.
#
# **Regla de oro:** el balanceo se aplica **solo al entrenamiento** (dentro de cada fold). Los datos de validación y de
# prueba deben conservar la distribución real; si no, las métricas no representan el uso real.

# %%
# 3.8.1 Comparación de estrategias de balanceo con un modelo de referencia (regresión logística, CV 5-fold)
def pipeline_referencia(muestreador=None, class_weight=None):
    """Preprocesamiento + (muestreador opcional) + regresión logística."""
    pasos = [("preprocesamiento", clone(preprocesador))]
    if muestreador is not None:
        pasos.append(("balanceo", muestreador))   # imblearn aplica el muestreo solo durante fit, nunca en predict
    pasos.append(("modelo", LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=RANDOM_STATE)))
    return ImbPipeline(pasos)


# Solo los muestreadores compiten para el pipeline final: sirven para los 8 modelos (class_weight no existe en KNN)
MUESTREADORES = {"Submuestreo aleatorio (RandomUnderSampler)": RandomUnderSampler(random_state=RANDOM_STATE),
                 "Sobremuestreo sintético (SMOTE)": SMOTE(random_state=RANDOM_STATE)}
estrategias = {"Sin balanceo": pipeline_referencia(),                                   # Línea base
               "Ponderación de clases (class_weight)": pipeline_referencia(class_weight="balanced"),
               **{nombre: pipeline_referencia(m) for nombre, m in MUESTREADORES.items()}}
filas = []
for nombre, pipe in estrategias.items():
    # cross_validate devuelve una métrica por fold; se reportan la media y la desviación del F1
    res = cross_validate(pipe, X_train, y_train, cv=CV, scoring=["f1", "precision", "recall", "roc_auc"], n_jobs=-1)
    filas.append({"estrategia": nombre, "f1": res["test_f1"].mean(), "f1_std": res["test_f1"].std(),
                  "precision": res["test_precision"].mean(), "recall": res["test_recall"].mean(),
                  "roc_auc": res["test_roc_auc"].mean()})
df_balanceo = guardar_tabla(pd.DataFrame(filas).round(4), "balanceo_comparacion")
df_balanceo

# %% [markdown]
# 🔎 **Interpretación:** Sin balanceo, la regresión logística tiene Precision 0,61 pero **Recall 0,21**: detecta solo uno de cada cinco estudiantes de alto desempeño, y su F1 es 0,316. Las tres estrategias de balanceo elevan el Recall a ~0,75 y el F1 a ~0,47:
#
# | Estrategia | F1 en CV |
# |---|---|
# | RandomUnderSampler | 0,4696 |
# | class_weight | 0,4685 |
# | SMOTE | 0,4673 |
#
# La **ROC-AUC casi no cambia** (~0,80): el balanceo no mejora la capacidad de ordenar a los estudiantes, solo desplaza el punto de decisión hacia la clase minoritaria. Es una lección clave: balancear es, en buena medida, elegir un umbral distinto.

# %%
# 3.8.2 Elección del muestreador (mejor F1) y efecto sobre la distribución de clases del entrenamiento
mejor_balanceo = df_balanceo[df_balanceo["estrategia"].isin(MUESTREADORES)].sort_values("f1", ascending=False).iloc[0]
BALANCEADOR = MUESTREADORES[mejor_balanceo["estrategia"]]              # Se usará en todos los pipelines de la fase 4
_, y_balanceado = clone(BALANCEADOR).fit_resample(X_train_cod, y_train)   # Solo para visualizar el efecto

fig, ejes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for eje, serie, titulo in zip(ejes, [y_train, pd.Series(y_balanceado)],
                              ["Entrenamiento original", f"Después de {mejor_balanceo['estrategia']}"]):
    conteo = serie.value_counts().sort_index()   # Índice 0 y 1 en orden
    eje.bar(["No alto (0)", "Alto (1)"], conteo.values, color=["#8C8C8C", "crimson"])
    for i, v in enumerate(conteo.values):
        eje.text(i, v, f"{v:,}", ha="center", va="bottom")   # Conteo sobre cada barra
    eje.set_title(titulo)
plt.tight_layout(); guardar_figura("balanceo_distribucion"); plt.show()

# Justificación para la tabla 3.7 del informe, con las cifras obtenidas
f1_sin_balanceo = df_balanceo.loc[df_balanceo["estrategia"] == "Sin balanceo", "f1"].iloc[0]
justificacion_balanceo = (
    f"{mejor_balanceo['estrategia']}: mejor F1 en CV entre los muestreadores ({mejor_balanceo['f1']:.4f} frente a "
    f"{f1_sin_balanceo:.4f} sin balanceo). Se aplica dentro del pipeline (imblearn), solo en los folds de entrenamiento, "
    "para no contaminar la validación ni el test.")
guardar_tabla(pd.DataFrame({"tecnica": [mejor_balanceo["estrategia"]], "justificacion": [justificacion_balanceo]}), "balanceo")
print(justificacion_balanceo)

# %% [markdown]
# 🔎 **Interpretación:** Se elige **RandomUnderSampler**, el muestreador con mejor F1. Su ventaja sobre SMOTE (0,0023) es mucho menor que la desviación estándar entre folds (~0,012): en la práctica las estrategias empatan.
# El submuestreo deja el entrenamiento balanceado 1:1 (≈3.600 casos por clase, frente a ≈17.400 negativos originales). Como ventaja adicional reduce el costo de cómputo: el SVM entrena con ~6 mil filas por fold en lugar de ~28 mil con SMOTE. Solo actúa en `fit` dentro de cada fold, así que validación y prueba conservan la distribución real.

# %% [markdown]
# ### 🧠 Para profundizar — Fase 3: Preparación de los datos
# **Ideas clave**
# - Las transformaciones **determinísticas** van antes de partir los datos; las **aprendidas** van dentro del pipeline.
# - El tipo de variable define su codificación: ordinal con orden explícito, nominal con One-Hot, numérica con escalado.
# - Los faltantes pueden ser informativos (MNAR): el indicador o la categoría `DESCONOCIDO` preservan esa señal.
# - Pearson ve relaciones lineales, Spearman monótonas y la información mutua cualquier dependencia.
# - Reducir dimensiones o balancear son **decisiones empíricas**: se comparan con y sin la técnica.
#
# **Preguntas de repaso**
# 1. ¿Qué pasaría con las métricas si se aplicara SMOTE **antes** de la validación cruzada?
# 2. ¿Por qué codificar `cole_jornada` como 0, 1, 2… sería un error para la regresión logística pero casi no afectaría a un árbol?
# 3. Si $\rho_{Spearman} = 0{,}40$ y $r_{Pearson} = 0{,}25$ entre estrato y puntaje, ¿qué se concluye sobre la forma de la relación?
# 4. ¿En qué casos sí convendría aplicar PCA en este problema?
#
# **Ejercicios**
# - Reemplaza SMOTE por `SMOTENC` (que trata las categóricas sin interpolarlas) y compara el F1 en CV.
# - Elimina `add_indicator=True` del pipeline ordinal y mide el efecto en el F1 de la regresión logística.
#
# **Referencias**
# - Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.
# - Little, R. J. A. y Rubin, D. B. (2019). *Statistical Analysis with Missing Data* (3.ª ed.). Wiley.
# - Kuhn, M. y Johnson, K. (2019). *Feature Engineering and Selection*. CRC Press.
# - Jolliffe, I. T. y Cadima, J. (2016). Principal component analysis: a review and recent developments. *Phil. Trans. R. Soc. A*, 374.
# - Chawla, N. V., Bowyer, K. W., Hall, L. O. y Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR*, 16, 321–357.
# - Cover, T. M. y Thomas, J. A. (2006). *Elements of Information Theory* (2.ª ed.). Wiley (cap. 2: información mutua).
# - Guía de usuario de scikit-learn: *Preprocessing data*, *Imputation of missing values* y *Column Transformer*.
