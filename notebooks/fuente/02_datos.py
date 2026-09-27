# %% [markdown]
# ## 2. Entendimiento de los datos
# ### 2.0 Carga de los datos desde Datos Abiertos Colombia

# %% [markdown]
# #### 📘 Concepto: Datos abiertos y API Socrata
# **Datos Abiertos Colombia** (datos.gov.co) publica conjuntos de datos del Estado con licencia libre; este usa
# CC BY-SA 4.0, que obliga a citar la fuente y a compartir obras derivadas con la misma licencia. La plataforma usa
# **Socrata**, que expone cada conjunto como una **API REST** (SODA) con un lenguaje de consulta parecido a SQL (**SoQL**):
#
# | Parámetro | Equivale en SQL a | Uso aquí |
# |---|---|---|
# | `$where` | `WHERE` | `periodo='20224' AND cole_depto_ubicacion='ANTIOQUIA'` |
# | `$select` | `SELECT` | (todas las columnas) |
# | `$limit` | `LIMIT` | 200.000, por encima del total esperado |
#
# **¿Por qué filtrar en el servidor?** El conjunto completo tiene millones de filas (2010–2022). Pedir solo el subconjunto
# necesario reduce la descarga de gigabytes a unos 75 MB. **Buena práctica:** guardar una copia local del dato crudo, porque
# los datos abiertos pueden actualizarse o dejar de estar disponibles, y el análisis debe poder repetirse con los mismos datos.

# %%
# 2.0.1 Descarga desde la API Socrata de datos.gov.co (o lectura de la copia local si ya existe)
ID_DATASET = "kgxf-xxbe"   # Identificador del conjunto "Resultados únicos Saber 11" en datos.gov.co
URL_DATASET = f"https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/{ID_DATASET}"   # Página pública (para citar)
FILTRO = "periodo='20224' AND cole_depto_ubicacion='ANTIOQUIA'"   # Cláusula SoQL: periodo 2022-2 y colegios de Antioquia
# urlencode convierte espacios, comillas y el símbolo $ a su forma segura dentro de una URL
URL_API = f"https://www.datos.gov.co/resource/{ID_DATASET}.csv?" + urlencode({"$where": FILTRO, "$limit": 200000})
RUTA_RAW = DIR_RAW / "saber11_antioquia_20224.csv.gz"   # Copia local comprimida con gzip (≈ 10 veces más liviana)

inicio = time.time()
if RUTA_RAW.exists():
    # dtype=str lee todo como texto: evita que pandas convierta códigos como "05001" en el número 5001
    df_raw = pd.read_csv(RUTA_RAW, dtype=str)
    origen = f"copia local ({RUTA_RAW.name})"
else:
    df_raw = pd.read_csv(URL_API, dtype=str)                  # pandas descarga el CSV directamente desde la URL
    df_raw.to_csv(RUTA_RAW, index=False, compression="gzip")  # Respaldo reproducible y entregable "dataset utilizado"
    origen = "API datos.gov.co"

print(f"Origen: {origen} | Filas: {df_raw.shape[0]:,} | Columnas: {df_raw.shape[1]} | {time.time() - inicio:.1f} s")
df_raw.head(3).T   # Transpuesta: con 51 columnas se leen mejor como filas

# %% [markdown]
# 🔎 **Interpretación:** Se leyó la copia local del CSV descargado de la API (144.766 filas × 51 columnas, en pocos segundos). La primera vez se descarga de datos.gov.co con el filtro SoQL `periodo='20224' AND cole_depto_ubicacion='ANTIOQUIA'`.
# En la vista transpuesta se ve la mezcla típica de un registro: códigos DANE como texto (`05034`), categorías socioeconómicas y los puntajes. Leer todo como texto (`dtype=str`) evitó que los códigos perdieran el cero inicial, como explica el 📘 de esta sección.

# %%
# 2.0.2 Conversión de tipos: los puntajes pasan a numérico; el resto son categóricas o texto
# Solo los puntajes son cantidades con las que tiene sentido operar (promedios, rangos); los códigos siguen siendo texto
COLS_PUNTAJES = ["punt_ingles", "punt_matematicas", "punt_sociales_ciudadanas",
                 "punt_c_naturales", "punt_lectura_critica", "punt_global"]
for col in COLS_PUNTAJES:
    # errors="coerce" convierte en NaN cualquier valor no numérico en lugar de detener la ejecución
    df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

# Resumen por columna: tipo, completitud y cardinalidad (primer diagnóstico de calidad)
resumen_tipos = pd.DataFrame({"tipo_pandas": df_raw.dtypes.astype(str),          # object = texto; float64 = numérico
                              "no_nulos": df_raw.notna().sum(),                  # Registros con dato
                              "pct_nulos": (df_raw.isna().mean() * 100).round(2),   # % de faltantes
                              "valores_unicos": df_raw.nunique()})               # Cardinalidad: 1 = constante
resumen_tipos

# %% [markdown]
# 🔎 **Interpretación:** Tras convertir los seis puntajes a numérico, el resumen revela los primeros problemas de calidad:
# - `estu_consecutivo` tiene solo **72.383 valores únicos en 144.766 filas**: cada evaluado aparece dos veces (se confirma en 2.2.2).
# - Hay nulos entre 3 % y 16 % en variables del hogar y del colegio (`cole_bilingue` 16,4 %).
# - `periodo`, `cole_depto_ubicacion` y `estu_estudiante` tienen un único valor: son constantes por construcción del filtro.
#
# La **cardinalidad** (valores únicos) anticipa qué columnas son identificadores (códigos de colegio con más de 1.000 valores) y cuáles son categóricas útiles (2 a 12 valores).

# %% [markdown]
# ### 2.1 Diccionario de datos

# %% [markdown]
# #### 📘 Concepto: Diccionario de datos y fuga de información
# Un **diccionario de datos** documenta cada variable: significado, tipo, dominio de valores y **rol** en el problema
# (entrada, salida o excluida). Obliga a entender cada columna antes de usarla y es la base de las reglas de calidad.
#
# La **fuga de información** (*data leakage*) ocurre cuando el modelo usa información que **no estaría disponible en el
# momento de predecir** o que contiene la respuesta de forma indirecta. Un modelo con fuga obtiene métricas excelentes en
# el laboratorio y falla en la práctica. Aquí, `punt_matematicas`, `punt_lectura_critica`, etc. son **componentes** del
# puntaje global: incluirlos haría el problema trivial y el modelo inútil para anticipar el resultado antes del examen.
# Kaufman et al. (2012) distinguen dos tipos:
# - **Fuga en las variables:** columnas que codifican la respuesta.
# - **Fuga en el entrenamiento:** usar datos de prueba al ajustar el preprocesamiento. Se evita con *pipelines* (fase 3).

# %%
# 2.1.1 Roles propuestos: variables de entrada (por tipo), objetivo y constantes de apoyo
# Separar por tipo es clave: cada tipo recibe un preprocesamiento distinto en la fase 3
VARIABLES_NUMERICAS = ["edad"]                                         # Cantidad continua: se imputa y estandariza
VARIABLES_ORDINALES = ["fami_estratovivienda", "fami_educacionmadre",  # Categorías con orden natural:
                       "fami_educacionpadre", "fami_personashogar",    # se codifican como 0 < 1 < 2 ...
                       "fami_cuartoshogar"]
VARIABLES_NOMINALES = ["estu_genero", "fami_tieneinternet", "fami_tienecomputador",   # Categorías sin orden:
                       "fami_tieneautomovil", "fami_tienelavadora", "cole_naturaleza",  # se codifican con One-Hot
                       "cole_jornada", "cole_area_ubicacion", "cole_bilingue", "cole_caracter",
                       "cole_genero", "cole_sede_principal", "cole_valle_aburra"]
FEATURES = VARIABLES_NUMERICAS + VARIABLES_ORDINALES + VARIABLES_NOMINALES   # 19 variables de entrada en total
OBJETIVO = "alto_desempeno"                                                 # Nombre de la variable a predecir
# Columnas originales que no entran al modelo, pero de las que se derivan variables nuevas
INSUMOS_DERIVADAS = {"estu_fechanacimiento": "edad", "cole_cod_mcpio_ubicacion": "cole_valle_aburra"}
# Códigos DANE de los 10 municipios del Valle de Aburrá (área metropolitana de Medellín)
CODIGOS_VALLE_ABURRA = ["05001", "05079", "05088", "05129", "05212",   # Medellín, Barbosa, Bello, Caldas, Copacabana
                        "05266", "05308", "05360", "05380", "05631"]   # Envigado, Girardota, Itagüí, La Estrella, Sabaneta

# Descripción de cada columna (según la documentación del ICFES) y de las variables derivadas
DESCRIPCIONES = {
    "periodo": "Periodo de aplicación (año + semestre; 20224 = 2022-2)",
    "estu_tipodocumento": "Tipo de documento de identidad del evaluado",
    "estu_consecutivo": "Identificador anonimizado del evaluado",
    "cole_area_ubicacion": "Área de ubicación de la sede (URBANO/RURAL)",
    "cole_bilingue": "Colegio bilingüe (S/N)",
    "cole_calendario": "Calendario académico del colegio (A/B/OTRO)",
    "cole_caracter": "Carácter del colegio (académico, técnico, técnico/académico)",
    "cole_cod_dane_establecimiento": "Código DANE del establecimiento",
    "cole_cod_dane_sede": "Código DANE de la sede",
    "cole_cod_depto_ubicacion": "Código DANE del departamento del colegio",
    "cole_cod_mcpio_ubicacion": "Código DANE del municipio del colegio",
    "cole_codigo_icfes": "Código ICFES del colegio",
    "cole_depto_ubicacion": "Departamento del colegio",
    "cole_genero": "Población del colegio por género (MIXTO/FEMENINO/MASCULINO)",
    "cole_jornada": "Jornada (MAÑANA, TARDE, COMPLETA, UNICA, NOCHE, SABATINA)",
    "cole_mcpio_ubicacion": "Municipio del colegio",
    "cole_naturaleza": "Naturaleza del colegio (OFICIAL/NO OFICIAL)",
    "cole_nombre_establecimiento": "Nombre del establecimiento",
    "cole_nombre_sede": "Nombre de la sede",
    "cole_sede_principal": "Sede principal (S/N)",
    "estu_cod_depto_presentacion": "Código del departamento donde presentó la prueba",
    "estu_cod_mcpio_presentacion": "Código del municipio donde presentó la prueba",
    "estu_cod_reside_depto": "Código del departamento de residencia",
    "estu_cod_reside_mcpio": "Código del municipio de residencia",
    "estu_depto_presentacion": "Departamento donde presentó la prueba",
    "estu_depto_reside": "Departamento de residencia",
    "estu_estadoinvestigacion": "Estado del resultado (PUBLICAR o en investigación)",
    "estu_estudiante": "Tipo de evaluado",
    "estu_fechanacimiento": "Fecha de nacimiento (dd/mm/aaaa)",
    "estu_genero": "Género del estudiante (F/M)",
    "estu_mcpio_presentacion": "Municipio donde presentó la prueba",
    "estu_mcpio_reside": "Municipio de residencia",
    "estu_nacionalidad": "Nacionalidad del estudiante",
    "estu_pais_reside": "País de residencia",
    "estu_privado_libertad": "Evaluado privado de la libertad (S/N)",
    "fami_cuartoshogar": "Número de cuartos del hogar",
    "fami_educacionmadre": "Máximo nivel educativo de la madre",
    "fami_educacionpadre": "Máximo nivel educativo del padre",
    "fami_estratovivienda": "Estrato socioeconómico de la vivienda",
    "fami_personashogar": "Número de personas en el hogar",
    "fami_tieneautomovil": "El hogar tiene automóvil (Si/No)",
    "fami_tienecomputador": "El hogar tiene computador (Si/No)",
    "fami_tieneinternet": "El hogar tiene internet (Si/No)",
    "fami_tienelavadora": "El hogar tiene lavadora (Si/No)",
    "desemp_ingles": "Nivel de desempeño en inglés (A-, A1, A2, B1, B+)",
    "punt_ingles": "Puntaje en inglés (0–100)",
    "punt_matematicas": "Puntaje en matemáticas (0–100)",
    "punt_sociales_ciudadanas": "Puntaje en sociales y ciudadanas (0–100)",
    "punt_c_naturales": "Puntaje en ciencias naturales (0–100)",
    "punt_lectura_critica": "Puntaje en lectura crítica (0–100)",
    "punt_global": "Puntaje global (0–500)",
    "edad": "DERIVADA: edad en años al 1-ago-2022 a partir de estu_fechanacimiento",
    "cole_valle_aburra": "DERIVADA: colegio en el Valle de Aburrá (Sí/No) según el código DANE del municipio",
    "alto_desempeno": "OBJETIVO: 1 si punt_global ≥ 300; 0 en otro caso",
}
# Motivo de exclusión de cada columna que no entra al modelo (documenta la decisión para el evaluador)
MOTIVOS_EXCLUSION = {
    "periodo": "Constante (filtro 2022-2)", "cole_depto_ubicacion": "Constante (filtro Antioquia)",
    "cole_cod_depto_ubicacion": "Constante (filtro Antioquia)", "estu_estudiante": "Constante",
    "cole_calendario": "Casi constante (calendario A)", "estu_nacionalidad": "Casi constante (Colombia)",
    "estu_pais_reside": "Casi constante (Colombia)", "estu_privado_libertad": "Casi constante (N)",
    "estu_tipodocumento": "Administrativa, sin relación causal plausible con el desempeño",
    "estu_consecutivo": "Identificador", "cole_codigo_icfes": "Identificador de alta cardinalidad",
    "cole_cod_dane_establecimiento": "Identificador de alta cardinalidad", "cole_cod_dane_sede": "Identificador de alta cardinalidad",
    "cole_nombre_establecimiento": "Identificador de alta cardinalidad", "cole_nombre_sede": "Identificador de alta cardinalidad",
    "cole_mcpio_ubicacion": "Redundante con el código de municipio",
    "estu_cod_depto_presentacion": "Redundante con la ubicación del colegio", "estu_cod_mcpio_presentacion": "Redundante con la ubicación del colegio",
    "estu_depto_presentacion": "Redundante con la ubicación del colegio", "estu_mcpio_presentacion": "Redundante con la ubicación del colegio",
    "estu_cod_reside_depto": "Redundante con la ubicación del colegio", "estu_cod_reside_mcpio": "Redundante con la ubicación del colegio",
    "estu_depto_reside": "Redundante con la ubicación del colegio", "estu_mcpio_reside": "Redundante con la ubicación del colegio",
    "estu_estadoinvestigacion": "Control de calidad: se usa para filtrar resultados no oficiales",
    "desemp_ingles": "Fuga de información: derivada del puntaje de inglés",
    "punt_ingles": "Fuga de información: componente del puntaje global", "punt_matematicas": "Fuga de información: componente del puntaje global",
    "punt_sociales_ciudadanas": "Fuga de información: componente del puntaje global",
    "punt_c_naturales": "Fuga de información: componente del puntaje global", "punt_lectura_critica": "Fuga de información: componente del puntaje global",
}


def rol_variable(var):
    """Rol de cada variable en el problema predictivo (se evalúa en orden de prioridad)."""
    if var == OBJETIVO:
        return "Salida (objetivo)"
    if var in FEATURES:
        return "Entrada"
    if var in INSUMOS_DERIVADAS:
        return f"Insumo para derivar {INSUMOS_DERIVADAS[var]}"
    if var == "punt_global":
        return "Insumo para construir el objetivo"   # No es entrada: de él se deriva la clase
    return "Excluida"


# El diccionario cubre las 51 columnas originales + las 2 derivadas + el objetivo
variables_diccionario = list(df_raw.columns) + ["edad", "cole_valle_aburra", OBJETIVO]
diccionario = pd.DataFrame({
    "variable": variables_diccionario,
    "descripcion": [DESCRIPCIONES[v] for v in variables_diccionario],
    "tipo": ["Num" if v in COLS_PUNTAJES + ["edad"] else "Cat" for v in variables_diccionario],   # Formato U4: Num/Cat
    "rol": [rol_variable(v) for v in variables_diccionario],
    "observacion": [MOTIVOS_EXCLUSION.get(v, "") for v in variables_diccionario],   # Vacío si la variable se usa
})
guardar_tabla(diccionario, "diccionario_datos")   # Insumo de la tabla 2.1 del informe
diccionario

# %% [markdown]
# 🔎 **Interpretación:** El diccionario documenta 54 variables: 51 originales, 2 derivadas y el objetivo. Para cada una registra descripción, tipo (Num/Cat), rol y motivo de exclusión.
# - Entran al modelo 19 variables: 1 numérica, 5 ordinales y 13 nominales.
# - Seis columnas se excluyen por **fuga de información**: los cinco puntajes por área y `desemp_ingles`, que son componentes del puntaje global.
# - El resto se descarta por ser identificadores, constantes o redundantes con la ubicación del colegio.
#
# Esta tabla es la tabla 2.1 del informe y el punto de partida de la selección de variables (3.1).

# %%
# 2.1.2 Verificación de las variables declaradas constantes o casi constantes (% de la categoría más frecuente)
# Una afirmación del diccionario ("casi constante") debe comprobarse con los datos, no suponerse
casi_constantes = ["periodo", "cole_depto_ubicacion", "estu_estudiante", "cole_calendario",
                   "estu_nacionalidad", "estu_pais_reside", "estu_privado_libertad"]
# value_counts(normalize=True) da proporciones; iloc[0] toma la categoría más frecuente (incluye NaN con dropna=False)
pd.Series({v: df_raw[v].value_counts(normalize=True, dropna=False).iloc[0] * 100 for v in casi_constantes},
          name="pct_categoria_mas_frecuente").round(2)

# %% [markdown]
# 🔎 **Interpretación:** La verificación confirma lo declarado en el diccionario: `periodo`, `cole_depto_ubicacion` y `estu_estudiante` son 100 % constantes. `cole_calendario` (99,7 % A), `estu_privado_libertad` (99,85 % N) y la nacionalidad y residencia (98,5 % Colombia) son casi constantes.
# Una variable con más del ~98 % en una sola categoría aporta casi nada a la discriminación y puede generar columnas one-hot con muy pocos casos. Excluirlas simplifica el modelo sin perder información. Es un ejemplo de selección **de filtro** (📘 de 3.1).

# %% [markdown]
# ### 2.2 Reglas de calidad

# %% [markdown]
# #### 📘 Concepto: Calidad de datos
# La calidad de datos se evalúa en varias **dimensiones** (Wang y Strong, 1996). Estas son las que se verifican aquí:
#
# | Dimensión | Pregunta | Regla en este proyecto |
# |---|---|---|
# | **Completitud** | ¿Falta el dato? | % de nulos por variable |
# | **Validez** | ¿El valor está en el dominio permitido? | Rangos numéricos (`punt_global ∈ [0, 500]`) y categorías válidas |
# | **Unicidad** | ¿Hay registros repetidos? | Filas duplicadas e identificador único |
# | **Consistencia** | ¿Los datos se contradicen? | Edad coherente con un estudiante de grado 11 (13–30 años) |
#
# Una **regla de calidad** se escribe *antes* de mirar el detalle de los datos (qué debería cumplirse) y luego se mide su
# **% de incumplimiento**. Así la limpieza de la fase 3 no es arbitraria: cada acción responde a una regla incumplida.

# %%
# 2.2.1 Definición de las reglas: rangos numéricos esperados y dominios categóricos válidos
def calcular_edad(fechas_texto):
    """Edad en años cumplidos a la fecha de referencia del examen (NaN si la fecha es inválida)."""
    # format fija el patrón dd/mm/aaaa; errors="coerce" convierte fechas imposibles (p. ej. 31/02) en NaT
    fecha = pd.to_datetime(fechas_texto, format="%d/%m/%Y", errors="coerce")
    # 365.25 días por año compensa los años bisiestos; floor da años cumplidos
    return np.floor((FECHA_REFERENCIA - fecha).dt.days / 365.25)


# Orden natural del nivel educativo (de menor a mayor): se reutiliza en la codificación ordinal de la fase 3
EDUCACION_ORDEN = ["Ninguno", "Primaria incompleta", "Primaria completa",
                   "Secundaria (Bachillerato) incompleta", "Secundaria (Bachillerato) completa",
                   "Técnica o tecnológica incompleta", "Técnica o tecnológica completa",
                   "Educación profesional incompleta", "Educación profesional completa", "Postgrado"]
# Regla de validez numérica: (mínimo, máximo) permitido según la escala del ICFES y la edad plausible
REGLAS_NUMERICAS = {"punt_global": (0, 500), "punt_matematicas": (0, 100), "punt_lectura_critica": (0, 100),
                    "punt_sociales_ciudadanas": (0, 100), "punt_c_naturales": (0, 100), "punt_ingles": (0, 100),
                    "edad": (13, 30)}
# Regla de validez categórica: dominio de valores aceptados por variable
CATEGORIAS_VALIDAS = {
    "estu_genero": ["F", "M"],
    "fami_estratovivienda": ["Sin Estrato", "Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"],
    "fami_educacionmadre": EDUCACION_ORDEN + ["No sabe", "No Aplica"],   # Se aceptan, pero se tratarán como faltantes
    "fami_educacionpadre": EDUCACION_ORDEN + ["No sabe", "No Aplica"],
    "fami_personashogar": ["1 a 2", "3 a 4", "5 a 6", "7 a 8", "9 o más"],
    "fami_cuartoshogar": ["Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis o mas"],
    "fami_tieneinternet": ["Si", "No"], "fami_tienecomputador": ["Si", "No"],
    "fami_tieneautomovil": ["Si", "No"], "fami_tienelavadora": ["Si", "No"],
    "cole_naturaleza": ["OFICIAL", "NO OFICIAL"],
    "cole_jornada": ["COMPLETA", "MAÑANA", "TARDE", "NOCHE", "SABATINA", "UNICA"],
    "cole_area_ubicacion": ["URBANO", "RURAL"],
    "cole_bilingue": ["S", "N"],
    "cole_caracter": ["ACADÉMICO", "TÉCNICO", "TÉCNICO/ACADÉMICO", "NO APLICA"],
    "cole_genero": ["MIXTO", "FEMENINO", "MASCULINO"],
    "cole_sede_principal": ["S", "N"],
}
edad_eda = calcular_edad(df_raw["estu_fechanacimiento"])   # Edad provisional solo para medir la regla

filas = []
for var, (minimo, maximo) in REGLAS_NUMERICAS.items():
    serie = edad_eda if var == "edad" else df_raw[var]
    # Las comparaciones con NaN dan False: los nulos se cuentan aparte, no como "fuera de rango"
    fuera = int(((serie < minimo) | (serie > maximo)).sum())
    filas.append({"variable": var, "rango_esperado": f"[{minimo}, {maximo}]", "nulos": int(serie.isna().sum()),
                  "fuera_de_rango": fuera, "pct_incumplimiento": round(100 * fuera / len(serie), 3)})
tabla_reglas_num = guardar_tabla(pd.DataFrame(filas), "reglas_numericas")   # Tabla 2.2 (numéricas) del informe
tabla_reglas_num

# %% [markdown]
# 🔎 **Interpretación:** Las reglas numéricas muestran que los puntajes respetan sus escalas: 0 % fuera de rango. Solo `punt_ingles` tiene 102 nulos.
# La regla de **consistencia** de la edad, en cambio, falla en 2.252 filas (1,56 %): edades fuera de [13, 30] que delatan fechas de nacimiento mal registradas (en 3.2.1 aparece incluso una edad mínima de 0 años). Estas filas no se eliminan: la edad se marca como faltante en 3.3 porque el resto del registro es válido.

# %%
# 2.2.2 Evaluación de dominios categóricos y de unicidad de registros
filas = []
for var, validas in CATEGORIAS_VALIDAS.items():
    serie = df_raw[var]
    invalidas = serie.notna() & ~serie.isin(validas)   # Hay dato, pero no pertenece al dominio válido
    filas.append({"variable": var, "categorias_validas": ", ".join(validas), "nulos": int(serie.isna().sum()),
                  "valores_invalidos": int(invalidas.sum()),
                  "ejemplos_invalidos": ", ".join(map(str, serie[invalidas].unique()[:5])),   # Para diagnosticar la causa
                  "pct_incumplimiento": round(100 * invalidas.mean(), 3)})
tabla_reglas_cat = guardar_tabla(pd.DataFrame(filas), "reglas_categoricas")   # Tabla 2.2 (categóricas) del informe

# Unicidad: un duplicado exacto repite las 51 columnas; un identificador repetido indica el mismo evaluado dos veces
print(f"Filas duplicadas exactas (51 columnas): {df_raw.duplicated().sum():,}")
print(f"Identificadores estu_consecutivo repetidos: {df_raw['estu_consecutivo'].duplicated().sum():,}")
print("Estado del resultado:", df_raw["estu_estadoinvestigacion"].value_counts().to_dict())   # Resultados no oficiales
tabla_reglas_cat

# %% [markdown]
# 🔎 **Interpretación:** - **Validez:** ninguna variable tiene valores fuera de su dominio (0 % de incumplimiento). No hacen falta homologaciones de texto.
# - **Unicidad:** hay **72.383 filas duplicadas exactas** (las 51 columnas iguales) y el mismo número de identificadores repetidos. La mitad del archivo publicado es una copia de la otra mitad: es un problema de la fuente, no del análisis.
# - **Resultados no oficiales:** 52 filas tienen estado "VALIDEZ OFICINA JURÍDICA" (26 estudiantes, porque también están duplicadas).
#
# Sin esta regla, el modelo se habría entrenado con datos repetidos y la validación sería optimista: el mismo estudiante podría caer en entrenamiento y en prueba.

# %% [markdown]
# ### 2.3 Distribución de la variable objetivo

# %% [markdown]
# #### 📘 Concepto: Variable objetivo y desbalance
# La variable objetivo se construye **discretizando** el puntaje global con un umbral de negocio:
#
# $$y_i = \mathbb{1}\left[\text{punt\_global}_i \ge 300\right]$$
#
# Si una clase es mucho menos frecuente que la otra, el problema está **desbalanceado**. Consecuencias:
# - Los algoritmos que minimizan el error global tienden a ignorar la clase minoritaria.
# - La *accuracy* deja de ser informativa: con 17 % de positivos, predecir siempre 0 da 83 % de acierto.
# - Hay que elegir métricas centradas en la clase positiva (Precision, Recall, F1, PR-AUC) y considerar técnicas de balanceo (fase 3).
#
# El **percentil** del umbral indica qué porcentaje de estudiantes queda por debajo. Sirve para saber si el umbral define un
# grupo "élite" muy pequeño o un grupo amplio.

# %%
# 2.3 Distribución del puntaje global y de la clase binaria derivada del umbral de 300
punt = df_raw["punt_global"].dropna()   # Se excluyen los nulos solo para describir la distribución
# kind="strict": % de estudiantes con puntaje estrictamente menor que 300
percentil_umbral = stats.percentileofscore(punt, UMBRAL_ALTO_DESEMPENO, kind="strict")
clase_eda = np.where(punt >= UMBRAL_ALTO_DESEMPENO, "Alto desempeño (1)", "No alto (0)")   # Discretización
conteo = pd.Series(clase_eda).value_counts().reindex(["No alto (0)", "Alto desempeño (1)"])   # Orden fijo para el gráfico

fig, ejes = plt.subplots(1, 2, figsize=(14, 4.5))
# Izquierda: histograma del puntaje continuo con el umbral marcado
sns.histplot(punt, bins=60, ax=ejes[0], color="#4C72B0")
ejes[0].axvline(UMBRAL_ALTO_DESEMPENO, color="crimson", ls="--",
                label=f"Umbral = {UMBRAL_ALTO_DESEMPENO} (percentil {percentil_umbral:.1f})")
ejes[0].set_title("Distribución del puntaje global"); ejes[0].set_xlabel("punt_global"); ejes[0].legend()
# Derecha: conteo de cada clase con su porcentaje (evidencia el desbalance)
ejes[1].bar(conteo.index, conteo.values, color=["#8C8C8C", "crimson"])
for i, v in enumerate(conteo.values):
    ejes[1].text(i, v, f"{v:,} ({v / conteo.sum():.1%})", ha="center", va="bottom")   # Etiqueta sobre cada barra
ejes[1].set_title("Variable objetivo: alto_desempeno")
plt.tight_layout(); guardar_figura("distribucion_objetivo"); plt.show()

# Asimetría > 0: cola larga hacia puntajes altos (pocos estudiantes con puntajes muy altos)
print(punt.describe().round(2).to_dict(), "| asimetría:", round(stats.skew(punt), 3))

# %% [markdown]
# 🔎 **Interpretación:** El puntaje global tiene media 245,9, mediana 242 y desviación 51,6, con una ligera asimetría positiva (0,31): pocos estudiantes alcanzan puntajes muy altos.
# El umbral de 300 queda en el **percentil 82,9**, así que el 17,1 % de los evaluados es "alto desempeño" (24.794 de 144.766 filas, antes de quitar duplicados). Es el **desbalance** anticipado en el 📘: un modelo que siempre dijera "no alto" acertaría el 82,9 % de las veces sin ninguna utilidad. Por eso la métrica principal es F1 y en 3.8 se evalúa el balanceo.

# %% [markdown]
# ### 2.4 Posibles sesgos en los datos

# %% [markdown]
# #### 📘 Concepto: Sesgos en los datos
# Un **sesgo** es una distorsión sistemática que hace que los datos o el modelo representen mal a ciertos grupos.
# Fuentes relevantes aquí (Suresh y Guttag, 2021):
#
# | Tipo | Qué es | Riesgo en este proyecto |
# |---|---|---|
# | **Representación** | Algunos grupos tienen pocos registros | Colegios rurales o de jornada nocturna subrepresentados |
# | **Selección** | Los datos no son una muestra aleatoria de la población | Solo aparecen quienes llegaron a grado 11 y presentaron el examen |
# | **Histórico** | Los datos reflejan desigualdades existentes | El modelo aprende la brecha socioeconómica y puede perpetuarla |
# | **Temporal** | Las condiciones cambian entre cohortes | 2022-2 es una cohorte pospandemia |
#
# Detectarlos antes de modelar permite interpretar bien los resultados. En la fase 5 se audita el desempeño del modelo por grupo.

# %%
# 2.4 Representación de grupos y tasa de alto desempeño por grupo (sesgo de representación e inequidad de base)
df_eda = df_raw.loc[df_raw["punt_global"].notna()].copy()   # Solo registros con objetivo definido
# Subregión: área metropolitana (Valle de Aburrá) frente al resto de Antioquia
df_eda["cole_valle_aburra"] = np.where(df_eda["cole_cod_mcpio_ubicacion"].isin(CODIGOS_VALLE_ABURRA),
                                       "Valle de Aburrá", "Resto de Antioquia")
df_eda["alto"] = (df_eda["punt_global"] >= UMBRAL_ALTO_DESEMPENO).astype(int)   # 1 = alto desempeño
GRUPOS_SESGO = {"estu_genero": "Género del estudiante", "cole_area_ubicacion": "Zona del colegio",
                "cole_naturaleza": "Naturaleza del colegio", "cole_valle_aburra": "Subregión",
                "fami_estratovivienda": "Estrato", "cole_jornada": "Jornada"}

filas = []
for var, titulo in GRUPOS_SESGO.items():
    # size = registros del grupo (representación); mean de una variable 0/1 = tasa de alto desempeño
    grupos = df_eda.groupby(df_eda[var].fillna("Sin dato"))["alto"].agg(["size", "mean"])
    for grupo, fila in grupos.iterrows():
        filas.append({"variable": titulo, "grupo": grupo, "registros": int(fila["size"]),
                      "pct_registros": 100 * fila["size"] / len(df_eda), "tasa_alto_desempeno_pct": 100 * fila["mean"]})
tabla_sesgos = guardar_tabla(pd.DataFrame(filas).round(2), "sesgos")

fig, ejes = plt.subplots(2, 3, figsize=(17, 8))
for eje, (titulo, sub) in zip(ejes.flat, tabla_sesgos.groupby("variable", sort=False)):   # Un panel por variable
    eje.bar(sub["grupo"].astype(str), sub["tasa_alto_desempeno_pct"], color="#4C72B0")
    eje.axhline(100 * df_eda["alto"].mean(), ls="--", color="k", lw=1)   # Tasa global como referencia
    eje.set_title(titulo); eje.set_ylabel("% alto desempeño"); eje.tick_params(axis="x", rotation=35)
plt.tight_layout(); guardar_figura("sesgos_por_grupo"); plt.show()
tabla_sesgos

# %% [markdown]
# 🔎 **Interpretación:** Las tasas se calculan sobre el archivo original. Los conteos están duplicados, pero los porcentajes no cambian, porque cada registro se repite exactamente dos veces. La tabla muestra **inequidades de base** marcadas:
# - **Estrato:** la tasa de alto desempeño pasa de 7,9 % (estrato 1) a 46,9 % (estrato 5). "Sin estrato" solo llega al 4,6 %.
# - **Naturaleza y subregión:** 31,5 % en colegios no oficiales frente a 13,5 % en oficiales; 21,2 % en el Valle de Aburrá frente a 11,3 % en el resto de Antioquia.
# - **Jornada:** en las jornadas nocturna y sabatina (jóvenes y adultos) la tasa es de apenas 2,1 %.
# - **Representación:** los colegios rurales (17 %) y las jornadas nocturna y sabatina (15 % entre ambas) son minoría.
#
# Estas diferencias son un **sesgo histórico** (📘): el modelo las aprenderá. Por eso en la fase 5 se audita su desempeño por grupo.

# %% [markdown]
# ### 🧠 Para profundizar — Fase 2: Entendimiento de los datos
# **Ideas clave**
# - Antes de transformar nada hay que **documentar** (diccionario) y **medir** (reglas de calidad) los datos.
# - La fuga de información es el error más costoso de un proyecto predictivo: se previene decidiendo el rol de cada variable.
# - El desbalance de la variable objetivo condiciona la métrica, el balanceo y la lectura de los resultados.
# - Los sesgos se identifican en los datos antes de que aparezcan amplificados en el modelo.
#
# **Preguntas de repaso**
# 1. ¿Por qué `punt_matematicas` es fuga de información y `fami_estratovivienda` no lo es?
# 2. ¿Qué diferencia hay entre una regla de validez y una de consistencia? Da un ejemplo de cada una con estos datos.
# 3. Si la tasa de alto desempeño es muy distinta entre colegios oficiales y no oficiales, ¿eso prueba que la naturaleza del colegio *causa* el desempeño?
#
# **Ejercicios**
# - Agrega una regla de consistencia: `punt_global` debería ser cercano a la combinación ponderada de los puntajes por área. Mide su incumplimiento.
# - Repite la tabla de sesgos para `cole_caracter` y comenta si hay grupos subrepresentados (< 1 % de los registros).
#
# **Referencias**
# - Wang, R. Y. y Strong, D. M. (1996). Beyond accuracy: What data quality means to data consumers. *Journal of Management Information Systems*, 12(4), 5–33.
# - Kaufman, S., Rosset, S., Perlich, C. y Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. *ACM TKDD*, 6(4).
# - Suresh, H. y Guttag, J. (2021). A framework for understanding sources of harm throughout the machine learning life cycle. *EAAMO '21*.
# - He, H. y Garcia, E. A. (2009). Learning from imbalanced data. *IEEE TKDE*, 21(9), 1263–1284.
# - Documentación de la API Socrata (SODA): https://dev.socrata.com/
