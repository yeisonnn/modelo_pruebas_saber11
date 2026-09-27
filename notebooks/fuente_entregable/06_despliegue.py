# %% [markdown]
# ## 6. Despliegue
# ### 6.0 Serialización del pipeline completo

# %% [markdown]
# **¿Qué hacemos y por qué?** Guardamos en disco el **pipeline completo** (preprocesamiento, balanceo y modelo) con su umbral
# de decisión. La aplicación usará exactamente lo que se evaluó.

# %%
#@copiar 6.1.1 como 6.0.1

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra que se serializó el pipeline completo (`preprocesamiento` → `balanceo` → `modelo`) en `models/pipeline_saber11.joblib`, con un tamaño de unos **20 KB**, porque una regresión logística guarda un coeficiente por columna codificada (un Random Forest de 341 árboles habría pesado decenas de MB).
# En general, el umbral de decisión (0,6152) se extrae del envoltorio `TunedThresholdClassifierCV` y se guarda en la metadata. El paso `balanceo` viaja dentro del pipeline, pero solo actúa en `fit`: al predecir no altera los datos.

# %% [markdown]
# **¿Qué hacemos y por qué?** Guardamos la **metadata** que usa la app: variables, categorías válidas, etiquetas, valores por
# defecto, umbral, métricas y versiones de las librerías.

# %%
#@copiar 6.1.2 como 6.0.2

# %% [markdown]
# 🔎 **Interpretación:** La salida muestra un extracto de la metadata que acompaña al modelo (`models/metadata.json`): el nombre del modelo final, el umbral (0,6152), el tipo de puntaje (probabilidad) y las métricas de prueba (F1 0,473, Recall 0,579, ROC-AUC 0,798). El archivo completo registra además:
# - Las 19 variables y sus categorías válidas, en el orden natural para las ordinales, y sus etiquetas legibles para el formulario.
# - Las versiones exactas de las librerías y el propósito académico del proyecto.
#
# En general, la aplicación lee todo desde este archivo, sin valores fijados en su código. `requiere_xgboost = False` indica que la app no necesita instalar xgboost, lo que aligera el despliegue.

# %% [markdown]
# **¿Qué hacemos y por qué?** Verificamos que el pipeline recargado desde disco produce exactamente las mismas predicciones.

# %%
#@copiar 6.1.3 como 6.0.3

# %% [markdown]
# 🔎 **Interpretación:** La salida confirma que el pipeline recargado desde disco produce **exactamente** los mismos puntajes en los 9.000 registros de prueba.
# En general, así se verifica que la serialización conserva el modelo completo (preprocesamiento incluido) y que lo evaluado en la fase 5 es lo que se despliega. La misma verificación está automatizada en `tests/test_modelo.py`, que además prueba categorías desconocidas y valores faltantes.

# %% [markdown]
# ### 6.1 Predicción con datos nuevos

# %% [markdown]
# **¿Qué hacemos y por qué?** Generamos 10 perfiles nuevos: 5 arquetipos diseñados y 5 aleatorios. Sirven como prueba de
# sensatez del modelo.

# %%
#@copiar 6.2.1 como 6.1.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra los 10 perfiles nuevos generados y guardados en `data/nuevos/perfiles_nuevos.csv`:
# - **Cinco arquetipos diseñados**, desde un estudiante rural de estrato 1 sin internet hasta uno de colegio bilingüe de estrato 6. El A5 tiene educación del padre "No sabe" para probar el manejo de faltantes.
# - **Cinco perfiles aleatorios**, muestreando cada variable de su distribución observada.
#
# En general, los arquetipos funcionan como **pruebas de sensatez**: si el modelo les asignara probabilidades contrarias al conocimiento del dominio, habría un error en el pipeline.

# %% [markdown]
# **¿Qué hacemos y por qué?** Predecimos e interpretamos los perfiles nuevos con el pipeline recargado (tabla 6.1 del informe).

# %%
#@copiar 6.2.2 como 6.1.2

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra la probabilidad y la predicción del modelo para cada perfil nuevo:
#
# | Perfil | Probabilidad | Predicción |
# |---|---|---|
# | A1 · rural, estrato 1 | 0,07 | No alto |
# | A2 · urbano oficial, estrato 2 | 0,42 | No alto, intermedio |
# | A3 · no oficial, estrato 4, padres profesionales | 0,91 | Alto |
# | A4 · bilingüe, estrato 6 | 0,87 | Alto |
# | A5 · adulto en jornada sabatina | 0,01 | No alto |
#
# En general, las predicciones son **coherentes con el dominio** y el pipeline procesó sin errores la respuesta "No sabe" del A5. Los perfiles aleatorios quedan entre 0,08 y 0,58 y ninguno supera el umbral de 0,615, algo esperable porque combinan al azar características de contextos distintos.

# %% [markdown]
# **¿Qué hacemos y por qué?** Medimos el desempeño en 1.000 estudiantes reales que **nunca** participaron en el modelado,
# para confirmar la generalización.

# %%
#@copiar 6.2.3 como 6.1.3

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra las métricas del modelo sobre 1.000 registros reales del pool, que no participaron en entrenamiento, validación ni prueba: **F1 = 0,495**, Recall 0,583 y Precision 0,430, muy cercanas a las de la prueba (0,473, 0,580 y 0,400).
# En general, esto confirma la **generalización**: la estimación de desempeño de la fase 5 era confiable, y la pequeña diferencia a favor del pool está dentro de la variabilidad esperable con 1.000 registros (≈170 positivos). Estos registros sirven también de ejemplo de predicción por lotes en la aplicación.

# %% [markdown]
# ### 6.2 Modelo prescriptivo: bandas de acompañamiento

# %% [markdown]
# **¿Qué hacemos y por qué?** Aplicamos la **regla de negocio 2** (sección 1.3) a los ~42 mil estudiantes que el modelo
# nunca vio. Cada uno recibe una banda (prioridad de nivelación, refuerzo focalizado o alto potencial). Validamos la regla
# comprobando que la tasa **real** de alto desempeño crece de banda en banda.

# %%
# 6.2.1 Regla prescriptiva: bandas de acompañamiento para los estudiantes del pool no visto
ORDEN_BANDAS = ["Prioridad de nivelación", "Refuerzo focalizado", "Alto potencial"]


def asignar_banda(probabilidad, umbral):
    """Regla 2 de 1.3: p ≥ t → alto potencial; t/2 ≤ p < t → refuerzo focalizado; p < t/2 → prioridad de nivelación."""
    return np.select([probabilidad >= umbral, probabilidad >= umbral / 2],
                     ["Alto potencial", "Refuerzo focalizado"], default="Prioridad de nivelación")


df_presc = df_pool.copy()                                                    # ~42 mil estudiantes fuera del modelado
df_presc["probabilidad"] = puntaje_positivo(pipeline_recargado, df_presc[FEATURES])
df_presc["banda"] = asignar_banda(df_presc["probabilidad"], umbral_final)
df_presc["municipio"] = df_raw.loc[df_presc.index, "cole_mcpio_ubicacion"]  # El índice conserva la fila original
df_presc["subregion"] = np.where(df_presc["cole_valle_aburra"] == "Sí", "Valle de Aburrá", "Resto de Antioquia")
resumen_bandas = (df_presc.groupby("banda")
                  .agg(estudiantes=("banda", "size"), probabilidad_media=("probabilidad", "mean"),
                       tasa_real_alto=(OBJETIVO, "mean"))
                  .reindex(ORDEN_BANDAS))
resumen_bandas["pct_estudiantes"] = 100 * resumen_bandas["estudiantes"] / len(df_presc)
resumen_bandas["tasa_real_alto"] = 100 * resumen_bandas["tasa_real_alto"]   # Validación: debe crecer de banda en banda
guardar_tabla(resumen_bandas.round(2).reset_index(), "e_prescriptivo_bandas")
resumen_bandas.round(2)

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra el resultado de aplicar la regla prescriptiva (regla 2 de 1.3) a los **42.357 estudiantes del pool**, que no participaron en el modelado. Con el umbral t = 0,615, las bandas son: prioridad de nivelación si p < 0,308; refuerzo focalizado si 0,308 ≤ p < 0,615; alto potencial si p ≥ 0,615.
#
# | Banda | Estudiantes | % | Probabilidad media | Tasa real de alto desempeño |
# |---|---|---|---|---|
# | 🔴 Prioridad de nivelación | 16.554 | 39,1 | 0,15 | 3,7 % |
# | 🟠 Refuerzo focalizado | 15.645 | 36,9 | 0,46 | 15,9 % |
# | 🟢 Alto potencial | 10.158 | 24,0 | 0,75 | 40,9 % |
#
# En general, las bandas **ordenan bien** a los estudiantes: la tasa real de alto desempeño se multiplica por 11 entre la primera y la tercera banda (3,7 % frente a 40,9 %). Por eso sirven para asignar tipos de acompañamiento distintos. Como las probabilidades están sobreestimadas (5.3.7), la probabilidad media de cada banda no debe leerse como su tasa esperada: lo que vale es el orden.

# %% [markdown]
# **¿Qué hacemos y por qué?** Con un **mapa de calor** vemos el % de estudiantes de cada banda según la combinación de
# naturaleza y zona del colegio: cada fila suma 100 % y los colores más oscuros señalan dónde se concentra cada banda.

# %%
# 6.2.2 Mapa de calor: % de estudiantes por banda según naturaleza y zona del colegio
tabla_calor = pd.crosstab([df_presc["cole_naturaleza"], df_presc["cole_area_ubicacion"]], df_presc["banda"],
                          normalize="index").mul(100)[ORDEN_BANDAS]          # % por fila: cada grupo suma 100
fig, eje = plt.subplots(figsize=(9, 4))
sns.heatmap(tabla_calor, annot=True, fmt=".1f", cmap="Blues", cbar=False, ax=eje)
eje.set_title("% de estudiantes por banda de acompañamiento"); eje.set_xlabel("Banda"); eje.set_ylabel("Naturaleza – zona")
plt.tight_layout(); guardar_figura("e_prescriptivo_calor"); plt.show()
# Tabla con los mismos porcentajes del mapa de calor
tabla_calor.round(1)

# %% [markdown]
# 🔎 **Interpretación:** El mapa de calor muestra, para cada combinación de naturaleza y zona del colegio, qué porcentaje de sus estudiantes cae en cada banda.
# En general:
# - **Oficial rural** concentra la mayor necesidad: el 72,6 % de sus estudiantes queda en prioridad de nivelación y solo el 3,4 % en alto potencial.
# - **Oficial urbano** es el grupo más grande y el más repartido: 45,5 % en refuerzo focalizado, 34,7 % en prioridad y 19,8 % en alto potencial.
# - **No oficial** tiene mayoría en alto potencial: 51,1 % en la zona urbana y 72,2 % en la rural. Este último es un grupo pequeño (unos 1.100 estudiantes en todo el conjunto limpio) de colegios privados campestres, con una tasa real de alto desempeño del 48 %.
#
# Para una secretaría de educación, esto sugiere concentrar los programas de nivelación en los colegios oficiales rurales y los de refuerzo en los oficiales urbanos.

# %% [markdown]
# **¿Qué hacemos y por qué?** Las **barras apiladas** muestran la composición de las bandas por subregión y por jornada, con
# el porcentaje dentro de cada tramo.

# %%
# 6.2.3 Barras apiladas: composición de las bandas por subregión y por jornada
fig, ejes = plt.subplots(1, 2, figsize=(16, 5))
for eje, var, titulo in [(ejes[0], "subregion", "subregión"), (ejes[1], "cole_jornada", "jornada")]:
    composicion = pd.crosstab(df_presc[var], df_presc["banda"], normalize="index").mul(100)[ORDEN_BANDAS]
    composicion = composicion.sort_values("Prioridad de nivelación")          # Grupos ordenados por prioridad
    composicion.plot.bar(stacked=True, ax=eje, color=["#dc2626", "#f59e0b", "#16a34a"], rot=0)
    for contenedor in eje.containers:
        eje.bar_label(contenedor, fmt="%.1f%%", label_type="center", fontsize=8)   # Etiqueta dentro de cada tramo
    eje.set_title(f"Bandas de acompañamiento por {titulo} (%)"); eje.set_xlabel(""); eje.set_ylabel("%")
    eje.legend(title="Banda", fontsize=8)
plt.tight_layout(); guardar_figura("e_prescriptivo_barras"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra, en barras apiladas, la composición de las bandas por subregión (izquierda) y por jornada (derecha).
# En general:
# - **Subregión:** fuera del Valle de Aburrá el 55,5 % de los estudiantes queda en prioridad de nivelación, el doble que dentro del Valle (27,8 %). En alto potencial, el Valle multiplica por 2,5 al resto (31,9 % frente a 12,5 %).
# - **Jornada:** la nocturna y la sabatina tienen más del 93 % de sus estudiantes en prioridad de nivelación y prácticamente ninguno en alto potencial. La completa tiene la mayor proporción de alto potencial (41,4 %), y la única y la de la mañana, la mayor de refuerzo focalizado (51,9 % y 46,3 %).
#
# Las jornadas de jóvenes y adultos requieren una estrategia propia: para ellas la banda es casi la misma para todos, así que el modelo aporta poca información individual y la decisión debe ser colectiva.

# %% [markdown]
# **¿Qué hacemos y por qué?** Identificamos los **10 municipios** con mayor % de estudiantes en "Prioridad de nivelación" (con
# al menos 150 estudiantes). Son los territorios donde convendría concentrar primero los programas de nivelación.

# %%
# 6.2.4 Top 10 municipios con mayor % de estudiantes en "Prioridad de nivelación" (≥ 150 estudiantes)
por_municipio = (df_presc.groupby("municipio")
                 .agg(estudiantes=("banda", "size"),
                      pct_prioridad=("banda", lambda b: 100 * (b == "Prioridad de nivelación").mean()),
                      pct_alto_potencial=("banda", lambda b: 100 * (b == "Alto potencial").mean()),
                      tasa_real_alto=(OBJETIVO, lambda s: 100 * s.mean())))           # Tasa real para contrastar
top_prioridad = por_municipio[por_municipio["estudiantes"] >= 150].nlargest(10, "pct_prioridad").round(1)   # Evita % inestables
guardar_tabla(top_prioridad.reset_index(), "e_prescriptivo_municipios")
print(f"Municipios evaluados (≥ 150 estudiantes en el pool): {(por_municipio['estudiantes'] >= 150).sum()}")
top_prioridad

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra los 10 municipios con mayor porcentaje de estudiantes en prioridad de nivelación, entre los 44 municipios con al menos 150 estudiantes en el pool.
# En general, en todos ellos al menos dos de cada tres estudiantes quedan en esa banda (del 67,8 % en El Bagre al 80,2 % en Anorí). Se concentran en el Bajo Cauca (Zaragoza, Tarazá, Cáceres, El Bagre), Urabá (San Pedro de Urabá, Necoclí, Arboletes, San Juan de Urabá) y el Nordeste (Anorí). Siete de ellos aparecen también entre los de menor tasa real en 2.4.6, y salvo Andes su tasa real de alto desempeño en el pool está entre 1,4 % y 6,3 %.
# **Andes** es la excepción instructiva: el modelo pone al 73,2 % de sus estudiantes en prioridad, pero su tasa real de alto desempeño es del 16,7 %, cercana a la media del departamento. Su perfil estructural (la mitad rural, 91 % oficial, 21 % en jornada sabatina y mayoría de estratos 1 y 2) predice resultados más bajos de los que sus estudiantes obtienen. Por eso las bandas deben **contrastarse con los resultados reales de cada municipio** antes de asignar recursos: el modelo hereda los sesgos de 5.4.

# %% [markdown]
# ### 6.3 Nivel de MLOps y diagrama de flujo de los componentes
# El modelo se ubica en el **nivel 1 de MLOps**:
# - **Automatizado y reproducible:** descarga por API, reglas de calidad verificadas con aserciones, pipeline único (preprocesamiento, balanceo y modelo), búsqueda de hiperparámetros con validación cruzada, serialización con metadata y pruebas automáticas (pytest) del notebook, del modelo y de la app.
# - **Manual:** el reentrenamiento con nuevas cohortes y el despliegue en Streamlit Cloud.
# - **Faltante:** integración continua y monitoreo del *drift* en producción, que serían el paso al nivel 2.

# %% [markdown]
# **¿Qué hacemos y por qué?** Dibujamos el diagrama de flujo de los componentes, desde el dato abierto hasta la aplicación web.

# %%
# 6.3.1 Diagrama de flujo de los componentes (del dato a la aplicación)
ETAPAS = ["Datos abiertos\n(API datos.gov.co)", "Calidad y limpieza\n(reglas + assert)", "Pipeline\n(prep + balanceo + modelo)",
          "Validación cruzada\ny ajuste (umbral)", "Serialización\n(joblib + metadata)", "App Streamlit\n(GitHub → Streamlit Cloud)"]
fig, eje = plt.subplots(figsize=(18, 3.2))
eje.set_xlim(0, len(ETAPAS) * 3); eje.set_ylim(0, 3); eje.axis("off")      # Lienzo sin ejes
ANCHO = 2.2                                                                 # Ancho de cada caja; deja 0,8 de espacio para la flecha
for i, texto in enumerate(ETAPAS):
    x = i * 3 + 0.3                                                         # Posición horizontal de cada caja
    eje.add_patch(FancyBboxPatch((x, 0.8), ANCHO, 1.4, boxstyle="round,pad=0.1", fc="#dbeafe", ec="#1d4ed8", lw=1.5))
    eje.text(x + ANCHO / 2, 1.5, texto, ha="center", va="center", fontsize=9.5)
    if i < len(ETAPAS) - 1:                                                 # Flecha en el hueco entre esta caja y la siguiente
        eje.annotate("", xy=(x + 3 - 0.15, 1.5), xytext=(x + ANCHO + 0.15, 1.5), zorder=3,
                     arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "#1d4ed8", "mutation_scale": 18})
eje.text(len(ETAPAS) * 1.5, 0.25, "Pruebas automáticas (pytest): notebook · modelo · app", ha="center", fontsize=10, style="italic")
eje.set_title("Flujo de componentes — nivel 1 de MLOps (reproducible, despliegue manual)", fontsize=13)
plt.tight_layout(); guardar_figura("e_flujo_mlops"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** El diagrama muestra los seis componentes de la solución en el orden en que fluye la información: datos abiertos → calidad y limpieza → pipeline → validación cruzada y ajuste del umbral → serialización → aplicación Streamlit. Las pruebas automáticas cubren el notebook, el modelo y la app.
# En general, todo el flujo es **reproducible** con un solo comando (descarga por API, semillas fijas y pipeline único), pero el reentrenamiento y el despliegue siguen siendo manuales. Eso ubica la solución en el **nivel 1 de MLOps**. El paso siguiente sería automatizar el reentrenamiento con cada nueva cohorte del ICFES y monitorear el *drift* de los datos.

# %% [markdown]
# ### 6.4 Aplicación web e información del despliegue
# - **App (`app.py`, Streamlit):** carga `models/pipeline_saber11.joblib` y `models/metadata.json`. Ofrece predicción
#   individual (formulario con un perfil típico por defecto), predicción por lotes (CSV o 1.000 registros de ejemplo) y una
#   pestaña con métricas, matriz de confusión e importancias. En ambos tipos de predicción muestra la **banda de
#   acompañamiento** de la regla 2 (sección 1.3), y en los lotes resume el porcentaje de registros en cada banda.
# - **Repositorio GitHub:** `app.py`, modelo serializado, `requirements.txt`, `README.md`, notebooks, datos y reporte HTML.
#
# | Información del despliegue | Estado |
# |---|---|
# | Modelo serializado | Sí — `models/pipeline_saber11.joblib` |
# | App Streamlit funcional | Sí — verificada en local y con pruebas automáticas |
# | Repositorio GitHub | [COMPLETAR: URL] |
# | URL pública en Streamlit.io | [COMPLETAR: URL] |
# | Enlace público de este cuaderno en Colab | [COMPLETAR: URL] |
