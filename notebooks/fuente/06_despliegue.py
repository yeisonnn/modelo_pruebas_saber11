# %% [markdown]
# ## 6. Despliegue
# ### 6.1 Serialización del pipeline completo

# %% [markdown]
# #### 📘 Concepto: Serialización de modelos
# **Serializar** es guardar en disco el objeto entrenado para usarlo después sin reentrenar. Aquí se serializa el
# **pipeline completo** (imputación → codificación → escalado → modelo) y no solo el modelo. Así la aplicación recibe datos
# en su forma original y les aplica **exactamente** las mismas transformaciones que en el entrenamiento; se evita el
# *training-serving skew* (Sculley et al., 2015).
#
# - **joblib** es el formato recomendado por scikit-learn: se basa en *pickle* y es eficiente con arreglos numéricos.
# - **Versiones:** un pickle solo es confiable con las **mismas versiones** de las librerías. Por eso se guardan en la metadata y se fijan en `requirements.txt`.
# - **Seguridad:** cargar un pickle ejecuta código. Solo se cargan archivos de **fuentes confiables** (aquí, el propio repositorio).
# - **Metadata:** variables esperadas, categorías válidas, umbral y métricas acompañan al modelo. La app se construye a partir de ella, sin valores fijos en el código.
#
# Las pruebas automáticas de `tests/test_modelo.py` verifican que el artefacto cargue, prediga y tolere valores inesperados.

# %%
# 6.1.1 Extraer el pipeline final y su umbral, y serializarlo con joblib
if isinstance(modelo_final, TunedThresholdClassifierCV):
    # El envoltorio del umbral contiene el pipeline reentrenado (estimator_) y el umbral óptimo (best_threshold_)
    pipeline_final, umbral_final = modelo_final.estimator_, float(modelo_final.best_threshold_)
else:
    pipeline_final = modelo_final
    umbral_final = 0.5 if hasattr(pipeline_final, "predict_proba") else 0.0   # 0 = frontera de decision_function

RUTA_MODELO = DIR_MODELOS / "pipeline_saber11.joblib"
joblib.dump(pipeline_final, RUTA_MODELO, compress=3)   # compress=3: buen equilibrio entre tamaño y velocidad de carga
tamano_mb = RUTA_MODELO.stat().st_size / 1e6
assert tamano_mb < 95, "El modelo supera el límite de GitHub (100 MB)"
print(f"Pipeline guardado: {RUTA_MODELO.name} ({tamano_mb:.1f} MB) | umbral de decisión = {umbral_final:.4f}")
if tamano_mb > 25:   # La carga por la web de GitHub admite hasta 25 MB; con git se admiten hasta 100 MB
    print("Aviso: supera 25 MB; súbalo con git (no con la carga web de GitHub).")
print("Pasos:", [nombre for nombre, _ in pipeline_final.steps])   # El balanceo se omite al predecir (solo actúa en fit)

# %% [markdown]
# 🔎 **Interpretación:** Se serializó el pipeline completo (`preprocesamiento` → `balanceo` → `modelo`) en `models/pipeline_saber11.joblib`. Pesa unos **20 KB** porque una regresión logística guarda un coeficiente por columna codificada. Un Random Forest de 341 árboles habría pesado decenas de MB.
# El umbral de decisión (0,6152) se extrajo del envoltorio `TunedThresholdClassifierCV` y se guardará en la metadata. El paso `balanceo` viaja dentro del pipeline, pero solo actúa en `fit`: al predecir no altera los datos.

# %%
# 6.1.2 Metadata para la app: variables, categorías válidas, etiquetas, umbral, métricas y versiones
# Etiquetas legibles para el formulario de la app (el usuario no ve nombres técnicos como fami_estratovivienda)
ETIQUETAS = {"edad": "Edad (años)", "estu_genero": "Género del estudiante", "fami_estratovivienda": "Estrato de la vivienda",
             "fami_educacionmadre": "Educación de la madre", "fami_educacionpadre": "Educación del padre",
             "fami_personashogar": "Personas en el hogar", "fami_cuartoshogar": "Cuartos en el hogar",
             "fami_tieneinternet": "¿Internet en el hogar?", "fami_tienecomputador": "¿Computador en el hogar?",
             "fami_tieneautomovil": "¿Automóvil en el hogar?", "fami_tienelavadora": "¿Lavadora en el hogar?",
             "cole_naturaleza": "Naturaleza del colegio", "cole_jornada": "Jornada", "cole_area_ubicacion": "Zona del colegio",
             "cole_bilingue": "¿Colegio bilingüe? (S/N)", "cole_caracter": "Carácter del colegio",
             "cole_genero": "Género del colegio", "cole_sede_principal": "¿Sede principal? (S/N)",
             "cole_valle_aburra": "¿Colegio en el Valle de Aburrá?"}
# Opciones de cada lista desplegable: categorías vistas en entrenamiento (nominales) u orden natural (ordinales)
categorias_app = {v: sorted(X_train[v].dropna().unique().tolist()) for v in VARIABLES_NOMINALES}
categorias_app.update({v: list(ORDEN_ORDINALES[v]) for v in VARIABLES_ORDINALES})
for v in ["fami_educacionmadre", "fami_educacionpadre"]:
    categorias_app[v] = categorias_app[v] + ["No sabe"]   # El pipeline lo trata como faltante (imputación)

metadata = {
    "proyecto": "Predicción de alto desempeño en Saber 11 — Antioquia 2022-2",
    "proposito": "Proyecto académico de maestría con fines exclusivamente educativos; no es una herramienta oficial",
    "fuente": URL_DATASET, "objetivo": f"alto_desempeno = 1 si punt_global >= {UMBRAL_ALTO_DESEMPENO}",
    "modelo_final": NOMBRE_FINAL, "umbral_decision": umbral_final,
    "tipo_puntaje": "probabilidad" if hasattr(pipeline_final, "predict_proba") else "decision",
    # Si el modelo es XGBoost, la app necesita instalar xgboost (se usa al generar requirements.txt)
    "requiere_xgboost": type(pipeline_final.named_steps["modelo"]).__module__.startswith("xgboost"),
    "features": FEATURES, "variables_numericas": VARIABLES_NUMERICAS, "variables_ordinales": VARIABLES_ORDINALES,
    "variables_nominales": VARIABLES_NOMINALES, "categorias": categorias_app, "etiquetas": ETIQUETAS,
    "rango_edad": [13, 30], "edad_mediana": float(X_train["edad"].median()),
    # Moda de cada variable: el formulario de la app arranca con un perfil típico y no con la primera opción alfabética
    "valores_por_defecto": {v: str(X_train[v].mode()[0]) for v in VARIABLES_ORDINALES + VARIABLES_NOMINALES},
    "metricas_test": {k: float(v) for k, v in df_test.set_index("modelo").loc[NOMBRE_FINAL].items()},   # float: JSON no admite numpy
    "n_entrenamiento": int(len(X_train)), "n_prueba": int(len(X_test)),
    "versiones": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__, "imbalanced-learn": imblearn.__version__,
                  "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    "fecha_entrenamiento": pd.Timestamp.now().strftime("%Y-%m-%d"),
}
# ensure_ascii=False conserva tildes y eñes legibles en el archivo JSON
(DIR_MODELOS / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: metadata[k] for k in ["modelo_final", "umbral_decision", "tipo_puntaje", "metricas_test"]},
                 ensure_ascii=False, indent=2))

# %% [markdown]
# 🔎 **Interpretación:** La metadata registra todo lo que la app necesita sin fijar valores en el código:
# - Las 19 variables y sus categorías válidas, en el orden natural para las ordinales.
# - Las etiquetas legibles para el formulario.
# - El umbral (0,615), el tipo de puntaje (probabilidad) y las métricas de test (F1 0,473, ROC-AUC 0,798).
# - Las versiones exactas de las librerías.
# - El propósito académico del proyecto.
#
# `requiere_xgboost = False`: la app no necesita instalar xgboost, lo que aligera el despliegue en Streamlit Cloud.

# %%
# 6.1.3 Verificación: el pipeline recargado desde disco produce exactamente los mismos puntajes
# joblib usa pickle: solo se carga el archivo que acabamos de generar (fuente confiable)
pipeline_recargado = joblib.load(RUTA_MODELO)
# np.allclose tolera diferencias mínimas de redondeo en coma flotante
assert np.allclose(puntaje_positivo(pipeline_recargado, X_test), puntaje_positivo(pipeline_final, X_test))
print("Pipeline recargado: predicciones idénticas en los", len(X_test), "registros de prueba")

# %% [markdown]
# 🔎 **Interpretación:** El pipeline recargado desde disco produce **exactamente** los mismos puntajes en los 9.000 registros de prueba. Así se verifica que la serialización conserva el modelo completo (preprocesamiento incluido) y que lo que se evaluó en la fase 5 es lo que se despliega.
# Esta misma verificación está automatizada en `tests/test_modelo.py`, que además prueba categorías desconocidas y valores faltantes.

# %% [markdown]
# ### 6.2 Predicción con datos nuevos

# %% [markdown]
# #### 📘 Concepto: Generalización a datos nuevos
# Un modelo es útil si **generaliza**: si mantiene su desempeño en datos que no participaron en ninguna decisión del
# entrenamiento. Aquí se prueban dos tipos de datos nuevos:
# - **Perfiles sintéticos:** combinaciones construidas a mano (arquetipos) o al azar. Sirven para verificar que el pipeline
#   acepta entradas nuevas y que las predicciones son **coherentes con el conocimiento del dominio** (prueba de sensatez).
# - **Registros reales no vistos:** ~42 mil registros del *pool* que quedaron fuera de la muestra de modelado. Si su F1 es
#   similar al del test, se confirma la estimación de desempeño.
#
# En un despliegue real además cambia la distribución de los datos con el tiempo (*data drift*: nuevas cohortes, cambios
# en el examen). Por eso los modelos en producción se **monitorean** y se reentrenan periódicamente.

# %%
# 6.2.1 Generación de 10 perfiles nuevos: 5 arquetipos diseñados + 5 aleatorios de las distribuciones observadas
# Perfil base "típico"; cada arquetipo modifica solo las variables que lo caracterizan
BASE_PERFIL = {"edad": 17, "estu_genero": "F", "fami_estratovivienda": "Estrato 2",
               "fami_educacionmadre": "Secundaria (Bachillerato) completa", "fami_educacionpadre": "Secundaria (Bachillerato) completa",
               "fami_personashogar": "3 a 4", "fami_cuartoshogar": "Tres", "fami_tieneinternet": "Si", "fami_tienecomputador": "Si",
               "fami_tieneautomovil": "No", "fami_tienelavadora": "Si", "cole_naturaleza": "OFICIAL", "cole_jornada": "MAÑANA",
               "cole_area_ubicacion": "URBANO", "cole_bilingue": "N", "cole_caracter": "ACADÉMICO", "cole_genero": "MIXTO",
               "cole_sede_principal": "S", "cole_valle_aburra": "Sí"}
ARQUETIPOS = [   # (nombre, cambios respecto al perfil base)
    ("A1 · Rural oficial, estrato 1, sin internet ni computador",
     {"edad": 18, "estu_genero": "M", "fami_estratovivienda": "Estrato 1", "fami_educacionmadre": "Primaria incompleta",
      "fami_educacionpadre": "Primaria incompleta", "fami_personashogar": "5 a 6", "fami_cuartoshogar": "Dos",
      "fami_tieneinternet": "No", "fami_tienecomputador": "No", "fami_tienelavadora": "No", "cole_jornada": "UNICA",
      "cole_area_ubicacion": "RURAL", "cole_caracter": "TÉCNICO/ACADÉMICO", "cole_sede_principal": "N", "cole_valle_aburra": "No"}),
    ("A2 · Urbano oficial del Valle de Aburrá, estrato 2, madre bachiller", {}),   # Es el perfil base sin cambios
    ("A3 · Urbano no oficial, estrato 4, padres profesionales",
     {"estu_genero": "M", "fami_estratovivienda": "Estrato 4", "fami_educacionmadre": "Educación profesional completa",
      "fami_educacionpadre": "Educación profesional completa", "fami_tieneautomovil": "Si", "cole_naturaleza": "NO OFICIAL",
      "cole_jornada": "COMPLETA"}),
    ("A4 · Colegio bilingüe no oficial, estrato 6, padres con posgrado",
     {"edad": 16, "fami_estratovivienda": "Estrato 6", "fami_educacionmadre": "Postgrado", "fami_educacionpadre": "Postgrado",
      "fami_cuartoshogar": "Cuatro", "fami_tieneautomovil": "Si", "cole_naturaleza": "NO OFICIAL", "cole_jornada": "COMPLETA",
      "cole_bilingue": "S"}),
    ("A5 · Adulto en jornada sabatina, estrato 2, padre desconocido",   # "No sabe" prueba el manejo de faltantes
     {"edad": 26, "fami_educacionmadre": "Primaria completa", "fami_educacionpadre": "No sabe", "fami_cuartoshogar": "Dos",
      "fami_tienecomputador": "No", "cole_naturaleza": "NO OFICIAL", "cole_jornada": "SABATINA"}),
]
rng = np.random.default_rng(RANDOM_STATE)   # Generador reproducible para los perfiles aleatorios
# {**BASE_PERFIL, **cambios}: el segundo diccionario sobrescribe las claves del primero
registros = [{"registro": nombre, **BASE_PERFIL, **cambios} for nombre, cambios in ARQUETIPOS]
for i in range(1, 6):
    # Cada variable se sortea de su distribución marginal observada (las combinaciones pueden ser inusuales)
    aleatorio = {v: rng.choice(X_train[v].dropna().to_numpy()) for v in FEATURES if v != "edad"}
    registros.append({"registro": f"R{i} · Perfil aleatorio", "edad": int(rng.integers(15, 20)), **aleatorio})
df_nuevos = pd.DataFrame(registros)[["registro"] + FEATURES]
df_nuevos.to_csv(DIR_NUEVOS / "perfiles_nuevos.csv", index=False)   # Entregable: "generar nuevos datos de entrada"
df_nuevos

# %% [markdown]
# 🔎 **Interpretación:** Se generaron 10 perfiles nuevos, guardados en `data/nuevos/perfiles_nuevos.csv`:
# - **Cinco arquetipos diseñados:** desde un estudiante rural de estrato 1 sin internet hasta uno de colegio bilingüe de estrato 6. El A5 tiene educación del padre "No sabe" para probar el manejo de faltantes.
# - **Cinco perfiles aleatorios**, muestreando cada variable de su distribución observada.
#
# Los arquetipos funcionan como **pruebas de sensatez**: si el modelo les asignara probabilidades contrarias al conocimiento del dominio, habría un error en el pipeline.

# %%
# 6.2.2 Predicción e interpretación de los perfiles nuevos con el pipeline recargado
def interpretar(puntaje, umbral):
    """Traduce el puntaje a un mensaje comprensible (tres bandas alrededor del umbral de decisión)."""
    if puntaje >= umbral:
        return "Perfil similar a estudiantes con alto desempeño: priorizar oferta de becas y estímulos."
    if puntaje >= umbral / 2:
        return "Probabilidad intermedia: puede alcanzar el umbral con refuerzo focalizado."
    return "Baja probabilidad de alto desempeño: perfil prioritario para nivelación y acompañamiento."


# Se usa el pipeline RECARGADO desde disco: es exactamente lo que usará la app
puntajes_nuevos = puntaje_positivo(pipeline_recargado, df_nuevos[FEATURES])
tabla_nuevos = pd.DataFrame({
    "registro": df_nuevos["registro"],
    "prediccion": [f"{'Alto desempeño' if s >= umbral_final else 'No alto'} (puntaje = {s:.3f})" for s in puntajes_nuevos],
    "interpretacion": [interpretar(s, umbral_final) for s in puntajes_nuevos]})
guardar_tabla(tabla_nuevos, "predicciones_nuevas")   # Tabla 6.1 del informe
tabla_nuevos

# %% [markdown]
# 🔎 **Interpretación:** Las predicciones son **coherentes con el dominio**:
#
# | Perfil | Probabilidad | Predicción |
# |---|---|---|
# | A1 · rural, estrato 1 | 0,07 | No alto |
# | A2 · urbano oficial, estrato 2 | 0,42 | No alto, intermedio |
# | A3 · no oficial, estrato 4, padres profesionales | 0,91 | Alto |
# | A4 · bilingüe, estrato 6 | 0,87 | Alto |
# | A5 · adulto en jornada sabatina | 0,01 | No alto |
#
# - El pipeline procesó sin errores la respuesta "No sabe" del A5.
# - Los perfiles aleatorios quedan entre 0,08 y 0,58. Ninguno supera el umbral de 0,615, lo que es esperable: combinan al azar características de contextos distintos.

# %%
# 6.2.3 Predicción sobre 1.000 registros reales que NUNCA se usaron (pool fuera de la muestra de modelado)
df_no_vistos = df_pool.sample(1000, random_state=RANDOM_STATE)
# Garantía explícita de que ninguno participó en entrenamiento, validación ni prueba
assert set(df_no_vistos.index).isdisjoint(df_modelo.index), "Hay registros usados en el modelado"
df_no_vistos[FEATURES + [OBJETIVO, "punt_global"]].to_csv(DIR_NUEVOS / "registros_no_vistos_1000.csv", index=False)   # Ejemplo para la app
pred_no_vistos = (puntaje_positivo(pipeline_recargado, df_no_vistos[FEATURES]) >= umbral_final).astype(int)
# Si las métricas del pool son parecidas a las del test, la estimación de desempeño es confiable
comparacion = pd.DataFrame({
    "conjunto": ["Prueba (30 %)", "Pool no visto (1.000)"],
    "f1": [metadata["metricas_test"]["f1"], f1_score(df_no_vistos[OBJETIVO], pred_no_vistos)],
    "recall": [metadata["metricas_test"]["recall"], recall_score(df_no_vistos[OBJETIVO], pred_no_vistos)],
    "precision": [metadata["metricas_test"]["precision"], precision_score(df_no_vistos[OBJETIVO], pred_no_vistos)]}).round(4)
comparacion

# %% [markdown]
# 🔎 **Interpretación:** Sobre 1.000 registros reales del pool, que no participaron en entrenamiento, validación ni prueba, el modelo obtiene **F1 = 0,495**, Recall 0,583 y Precision 0,430. Son cifras muy cercanas a las del test (0,473, 0,580 y 0,400).
# Esto confirma la **generalización**: la estimación de desempeño de la fase 5 era confiable. La pequeña diferencia a favor del pool está dentro de la variabilidad esperable con 1.000 registros (≈170 positivos). Estos registros se usan también como ejemplo de predicción por lotes en la app.

# %% [markdown]
# ### 6.3 Aplicación web y publicación
# - La app `app.py` (Streamlit) carga `models/pipeline_saber11.joblib` y `models/metadata.json`. Ofrece predicción
#   individual, predicción por lotes (CSV) y una pestaña con métricas, matriz de confusión e importancias. En cada
#   predicción indica la banda de acompañamiento (prioridad de nivelación si p < t/2, refuerzo focalizado si
#   t/2 ≤ p < t, alto potencial si p ≥ t). Muestra un aviso de que es un **proyecto académico con fines educativos**.
# - Repositorio GitHub: `app.py`, modelo serializado, `requirements.txt`, `README.md`, notebook, datos y reporte HTML.
# - Despliegue: share.streamlit.io → *Create app* → repositorio → `app.py` → Python 3.12.
#
# **¿Por qué Streamlit?** Convierte un script de Python en una aplicación web sin programar HTML ni JavaScript. Cada
# interacción vuelve a ejecutar el script, y `st.cache_resource` evita recargar el modelo en cada clic. Es suficiente para
# demostraciones y prototipos. Un servicio de producción usaría una API (p. ej. FastAPI) con monitoreo.

# %% [markdown]
# ### 🧠 Para profundizar — Fase 6: Despliegue
# **Ideas clave**
# - Se despliega el **pipeline completo** y su **metadata**, no solo el modelo: así lo que se evalúa es lo que se usa.
# - La reproducibilidad exige fijar versiones; la seguridad exige cargar pickles solo de fuentes confiables.
# - Los datos nuevos (sintéticos y reales no vistos) comprueban la coherencia y la generalización del modelo.
# - Desplegar no es el final: en un sistema real habría que monitorear el *drift* y reentrenar.
#
# **Preguntas de repaso**
# 1. ¿Qué error ocurriría si la app aplicara su propio escalado en lugar de usar el del pipeline serializado?
# 2. ¿Por qué el umbral de decisión se guarda en la metadata en lugar de escribirse directamente en `app.py`?
# 3. ¿Qué señales indicarían que el modelo debe reentrenarse con la cohorte 2023?
#
# **Ejercicios**
# - Agrega a la app un gráfico que muestre cómo cambia la probabilidad al variar solo el estrato de un perfil (análisis de sensibilidad).
# - Descarga el periodo 2021-2 (`periodo='20214'`) de la API y evalúa el modelo en esa cohorte: ¿hay *drift*?
#
# **Referencias**
# - Sculley, D. et al. (2015). Hidden technical debt in machine learning systems. *NeurIPS*.
# - Documentación de scikit-learn: *Model persistence* (https://scikit-learn.org/stable/model_persistence.html).
# - Documentación de Streamlit: https://docs.streamlit.io/
# - Huyen, C. (2022). *Designing Machine Learning Systems*. O'Reilly (caps. 7 y 8: despliegue y monitoreo).

# %% [markdown]
# ## Conclusiones
# 1. **Datos:** el conjunto 2022-2 de Antioquia (ICFES, Datos Abiertos Colombia) venía con el 50 % de filas duplicadas. La limpieza dejó 72.357 estudiantes con 17,1 % de alto desempeño (≥ 300 puntos). Las reglas de calidad y las pruebas con `assert` fueron esenciales para detectarlo.
# 2. **Modelo:** la **Regresión Logística con umbral ajustado (0,615)** logró F1 = 0,473 (IC95 % 0,453–0,494), Recall 0,58 y ROC-AUC 0,80 en test, y F1 = 0,495 en 1.000 registros nunca vistos.
# 3. **Comparación:** ni los modelos flexibles (Random Forest, SVM) ni los ensambles (votación, bagging, XGBoost) superaron al modelo lineal. Las pruebas estadísticas mostraron equivalencia en F1, así que se eligió el modelo más simple e interpretable. La mejora más grande vino de **ajustar el umbral**, no del algoritmo.
# 4. **Factores asociados:** la edad (extraedad), la jornada, la educación de los padres, el género, el acceso a computador y la naturaleza del colegio son las variables con mayor peso. Son asociaciones, no causas.
# 5. **Sesgo:** el modelo funciona peor en colegios oficiales (F1 0,34) y fuera del Valle de Aburrá (0,38), y sobrepredice alto desempeño en hombres. Reproduce las desigualdades del sistema educativo. Por eso su uso, aun académico, exige cautela y nunca debe servir para excluir.
# 6. **Limitaciones:** hay una sola cohorte (2022-2, pospandemia) y solo variables socioeconómicas y del colegio. La curva de aprendizaje muestra alto sesgo: con estas variables el techo de F1 ronda 0,47–0,49.
# 7. **Trabajo futuro:** incorporar el **efecto colegio** (promedio histórico del colegio en cohortes anteriores, extensión E1), validar en otras cohortes para medir *drift*, calibrar las probabilidades y explorar métricas de equidad como restricción del entrenamiento.
