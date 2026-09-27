# %% [markdown]
# ## 4. Modelamiento

# %% [markdown]
# #### 📘 Concepto: Validación cruzada y pipelines
# **Validación cruzada k-fold:** el entrenamiento se divide en $k$ partes (*folds*). Se entrena $k$ veces, cada vez con
# $k-1$ folds, y se evalúa en el restante. La estimación del desempeño es el promedio de las $k$ evaluaciones, y su
# desviación estándar indica la **estabilidad** del modelo. Con `StratifiedKFold` cada fold conserva la proporción de clases.
#
# $$\widehat{F1}_{CV} = \frac{1}{k}\sum_{j=1}^{k} F1_j \qquad (k = 5)$$
#
# **Pipeline:** encadena preprocesamiento → balanceo → modelo en un solo objeto. En cada fold, *todo* se ajusta solo con
# los folds de entrenamiento, lo que elimina la fuga de información por preprocesamiento (Kaufman et al., 2012). Además, el
# pipeline completo es lo que se serializa en la fase 6: la app recibe datos crudos y aplica exactamente las mismas
# transformaciones.
#
# Todos los modelos usan **los mismos 5 folds**, así que la comparación entre modelos es **pareada** (se usa en la fase 5).

# %%
# 4.0 Fábrica de pipelines y función de validación cruzada común
def crear_pipeline(modelo):
    """Preprocesamiento + (PCA opcional) + balanceo + modelo; todo se ajusta dentro de cada fold."""
    pasos = [("preprocesamiento", clone(preprocesador))]   # clone: cada pipeline recibe su propio preprocesador sin entrenar
    if APLICAR_PCA:                                         # Decisión tomada con evidencia en la prueba de PCA (fase 3)
        pasos.append(("pca", PCA(n_components=0.90, random_state=RANDOM_STATE)))
    pasos += [("balanceo", clone(BALANCEADOR)), ("modelo", modelo)]   # Muestreador elegido al comparar estrategias de balanceo (fase 3)
    return ImbPipeline(pasos)


METRICAS_CV = ["f1", "precision", "recall", "accuracy", "roc_auc"]   # F1 es la principal; las demás dan contexto


def evaluar_cv(nombre, modelo):
    """CV estratificada 5-fold sobre el entrenamiento; devuelve el resumen y el F1 de cada fold."""
    inicio = time.time()
    # n_jobs=-1 entrena los 5 folds en paralelo con todos los núcleos disponibles
    res = cross_validate(crear_pipeline(modelo), X_train, y_train, cv=CV, scoring=METRICAS_CV, n_jobs=-1)
    resumen = {"modelo": nombre, **{f"{m}_mean": res[f"test_{m}"].mean() for m in METRICAS_CV},
               "f1_std": res["test_f1"].std(), "tiempo_s": time.time() - inicio}   # std del F1 = estabilidad
    return resumen, res["test_f1"]   # También el F1 por fold, para las pruebas estadísticas pareadas


# Se muestran los pasos del pipeline para verificar el orden
print("Pasos del pipeline:", [nombre for nombre, _ in crear_pipeline(LogisticRegression()).steps])

# %% [markdown]
# 🔎 **Interpretación:** El pipeline de todos los modelos tiene tres pasos: `preprocesamiento` → `balanceo` → `modelo`. No hay paso `pca`, por la decisión de 3.7.2.
# Con esta fábrica, cada modelo se evalúa exactamente igual: mismos folds, mismo preprocesamiento aprendido dentro de cada fold y mismo muestreador. Cualquier diferencia de F1 se debe al algoritmo y no a cómo se prepararon los datos.

# %% [markdown]
# ### 4.1 Modelos clásicos

# %% [markdown]
# #### 📘 Concepto: Modelos clásicos
# | Modelo | Idea central | Hiperparámetro clave | Fortalezas | Debilidades |
# |---|---|---|---|---|
# | **Regresión Logística** | $P(y=1\mid x) = \sigma(w^\top x + b) = \frac{1}{1+e^{-(w^\top x+b)}}$; minimiza la log-pérdida con regularización | `C` (inverso de la regularización) | Interpretable (coeficientes = log-odds), rápida, probabilidades calibradas | Frontera lineal |
# | **KNN** | Clase mayoritaria entre los $k$ vecinos más cercanos (distancia de Minkowski) | `n_neighbors`, `weights` | Sin supuestos de forma, frontera flexible | Lento al predecir, sensible a la escala y a la dimensión |
# | **Árbol de Decisión** | Divisiones binarias que reducen la impureza: Gini $=1-\sum_c p_c^2$ o entropía $=-\sum_c p_c\log p_c$ | `max_depth`, `min_samples_leaf` | Muy interpretable, captura interacciones | Alta varianza (sobreajusta) |
# | **SVM (RBF)** | Hiperplano de **margen máximo** en un espacio transformado por el kernel $K(x,x') = e^{-\gamma\lVert x-x'\rVert^2}$ | `C`, `gamma` | Fronteras no lineales complejas | Costo $O(n^2)$–$O(n^3)$; no da probabilidades por defecto |
# | **Random Forest** | Promedio de muchos árboles entrenados con *bootstrap* y un subconjunto aleatorio de variables en cada división | `n_estimators`, `max_features` | Robusto, poca varianza, pocas decisiones de ajuste | Menos interpretable, modelo pesado |
#
# El dilema **sesgo-varianza** organiza la comparación: los modelos simples (logística) tienen más sesgo; los flexibles
# (árbol profundo, KNN con $k$ pequeño) tienen más varianza. La validación cruzada muestra cuál equilibra mejor en este problema.

# %%
# 4.1.1 Cinco modelos clásicos con hiperparámetros iniciales razonables
modelos_clasicos = {
    "Regresión Logística": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),   # max_iter alto: asegura convergencia
    "KNN": KNeighborsClassifier(n_neighbors=15),                                           # k impar evita empates
    "Árbol de Decisión": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),   # Profundidad acotada contra sobreajuste
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE),       # Valores por defecto de scikit-learn
    # n_jobs=1 dentro del modelo: el paralelismo ya lo aporta la validación cruzada (evita saturar la CPU)
    "Random Forest": RandomForestClassifier(n_estimators=200, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=1),
}
# Texto para la tabla 4.1 del informe
HIPERPARAMETROS_INICIALES = {
    "Regresión Logística": "C=1.0, penalización L2, max_iter=2000",
    "KNN": "n_neighbors=15, weights='uniform', p=2 (euclidiana)",
    "Árbol de Decisión": "max_depth=8, criterion='gini', min_samples_leaf=1",
    "SVM (RBF)": "kernel='rbf', C=1.0, gamma='scale'",
    "Random Forest": "n_estimators=200, min_samples_leaf=5, max_features='sqrt'",
}
resultados_cv, f1_por_fold = [], {}   # Resúmenes por modelo y F1 de cada fold (para comparaciones pareadas)
for nombre, modelo in modelos_clasicos.items():
    resumen, f1s = evaluar_cv(nombre, modelo)
    resultados_cv.append(resumen)
    f1_por_fold[nombre] = f1s
    print(f"{nombre:<22} F1 = {resumen['f1_mean']:.4f} ± {resumen['f1_std']:.4f}  ({resumen['tiempo_s']:.0f} s)")

df_clasicos = pd.DataFrame(resultados_cv).sort_values("f1_mean", ascending=False).reset_index(drop=True)   # Ranking
guardar_tabla(pd.DataFrame({"modelo": df_clasicos["modelo"],
                            "hiperparametros_iniciales": df_clasicos["modelo"].map(HIPERPARAMETROS_INICIALES),
                            "metrica": "F1 (clase positiva)", "cv_mean": df_clasicos["f1_mean"].round(4),
                            "cv_std": df_clasicos["f1_std"].round(4)}), "modelos_clasicos")   # Tabla 4.1 del informe
df_clasicos.round(4)

# %% [markdown]
# 🔎 **Interpretación:** Ranking de los modelos clásicos por F1 en validación cruzada:
#
# | Modelo | F1 en CV |
# |---|---|
# | Regresión Logística | 0,4696 ± 0,012 |
# | Random Forest | 0,4690 ± 0,010 |
# | SVM RBF | 0,4659 ± 0,013 |
# | Árbol de decisión | 0,4298 |
# | KNN | 0,4289 |
#
# - **Los tres primeros empatan en la práctica:** sus diferencias son mucho menores que la desviación entre folds, y la ROC-AUC de los tres ronda 0,80. Que el modelo **lineal** empate con los flexibles indica que la señal disponible es esencialmente aditiva: la flexibilidad extra no encuentra patrones adicionales.
# - **El árbol y KNN quedan claramente atrás:** el árbol individual tiene alta varianza, y KNN sufre con 50 dimensiones en su mayoría binarias.
# - **Costo:** el SVM es, con diferencia, el más lento (unos 25 s frente a 2 s de la logística), aun con el submuestreo.

# %%
# 4.1.2 Variabilidad del F1 entre folds (misma partición para todos: comparación pareada)
# Un diagrama de caja muestra mediana, dispersión y folds atípicos: dos modelos con cajas solapadas pueden no diferir realmente
fig, eje = plt.subplots(figsize=(10, 4))
sns.boxplot(data=pd.DataFrame(f1_por_fold)[df_clasicos["modelo"]], ax=eje, color="#9DB7D5")   # Orden del ranking
eje.set_ylabel("F1 por fold"); eje.set_title("Modelos clásicos: F1 en validación cruzada (5 folds)")
plt.tight_layout(); guardar_figura("cv_clasicos"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** El diagrama de cajas confirma lo anterior: las cajas de la logística, Random Forest y el SVM se solapan casi por completo, mientras las del árbol y KNN están claramente más abajo.
# Mirar la **distribución por fold**, y no solo la media, evita conclusiones apresuradas: con desviaciones de ~0,01, una diferencia de medias de 0,0006 (logística frente a Random Forest) es ruido. Esa intuición se formaliza con el t-test corregido en la fase 5.

# %% [markdown]
# ### 4.2 Modelos de ensamble (actividad investigativa)

# %% [markdown]
# #### 📘 Concepto: Métodos de ensamble
# Un **ensamble** combina varios modelos para obtener uno mejor que cada componente. Hay tres familias:
#
# - **Votación** (*voting*): modelos **distintos** predicen y se combinan. En el voto **duro** gana la clase más votada; en el
#   voto **suave** se promedian las probabilidades, $\hat p(x) = \sum_m w_m\,\hat p_m(x) / \sum_m w_m$. Funciona mejor
#   cuando los modelos cometen **errores distintos** (baja correlación).
# - **Bagging** (Breiman, 1996): muchos modelos **iguales** (aquí árboles profundos) entrenados en muestras *bootstrap*
#   (con reemplazo) y promediados. Promediar $B$ modelos con varianza $\sigma^2$ y correlación $\rho$ da varianza
#   $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$: **reduce la varianza**. Random Forest es bagging más un sorteo de variables
#   en cada división, que baja $\rho$.
# - **Boosting** (Friedman, 2001): modelos débiles (árboles poco profundos) entrenados **en secuencia**; cada uno ajusta el
#   gradiente del error de los anteriores, $F_m(x) = F_{m-1}(x) + \eta\,h_m(x)$. **Reduce el sesgo**. **XGBoost** (Chen y
#   Guestrin, 2016) agrega regularización L1/L2 sobre las hojas, submuestreo y un algoritmo eficiente por histogramas.

# %%
# 4.2.1 Votación suave (3 mejores clásicos con probabilidades), Bagging de árboles y XGBoost
# Alias cortos: VotingClassifier exige nombres simples para sus modelos internos
ALIAS = {"Regresión Logística": "logistica", "KNN": "knn", "Árbol de Decisión": "arbol",
         "SVM (RBF)": "svm", "Random Forest": "rf"}
# El voto suave necesita predict_proba: el SVC sin probability=True no lo tiene y queda excluido automáticamente
con_probabilidad = [n for n in df_clasicos["modelo"] if hasattr(modelos_clasicos[n], "predict_proba")]
TOP3_VOTO = con_probabilidad[:3]   # Los tres mejores clásicos según el ranking de 4.1
modelos_ensamble = {
    "Votación (soft)": VotingClassifier([(ALIAS[n], clone(modelos_clasicos[n])) for n in TOP3_VOTO], voting="soft"),
    # Bagging de árboles sin límite de profundidad (alta varianza, que el promedio reduce); max_samples = 80 % por árbol
    "Bagging (árboles)": BaggingClassifier(estimator=DecisionTreeClassifier(random_state=RANDOM_STATE), n_estimators=100,
                                           max_samples=0.8, random_state=RANDOM_STATE, n_jobs=1),
    # Boosting: árboles poco profundos (max_depth=4) + tasa de aprendizaje moderada + submuestreo de filas y columnas
    "XGBoost": XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                             eval_metric="logloss", tree_method="hist", random_state=RANDOM_STATE, n_jobs=1),
}
resultados_ens = []
for nombre, modelo in modelos_ensamble.items():   # Misma evaluación y mismos folds que los clásicos
    resumen, f1s = evaluar_cv(nombre, modelo)
    resultados_ens.append(resumen)
    f1_por_fold[nombre] = f1s
    print(f"{nombre:<22} F1 = {resumen['f1_mean']:.4f} ± {resumen['f1_std']:.4f}  ({resumen['tiempo_s']:.0f} s)")
df_ensambles = pd.DataFrame(resultados_ens).sort_values("f1_mean", ascending=False).reset_index(drop=True)

mejor_clasico = df_clasicos.iloc[0]   # Referencia para medir la mejora de cada ensamble
TIPOS_ENSAMBLE = {"Votación (soft)": "Votación suave (heterogéneo): " + ", ".join(TOP3_VOTO),
                  "Bagging (árboles)": "Bagging (homogéneo, paralelo, bootstrap)",
                  "XGBoost": "Boosting (secuencial, por gradiente)"}
guardar_tabla(pd.DataFrame({   # Tabla 4.2 del informe
    "modelo": df_ensambles["modelo"], "tipo": df_ensambles["modelo"].map(TIPOS_ENSAMBLE),
    "desempeno": [f"F1 CV = {m:.4f} ± {s:.4f}" for m, s in zip(df_ensambles["f1_mean"], df_ensambles["f1_std"])],
    "mejora_vs_clasico": [f"{d:+.4f} vs {mejor_clasico['modelo']}" for d in df_ensambles["f1_mean"] - mejor_clasico["f1_mean"]],
}), "modelos_ensamble")
df_ensambles.round(4)

# %% [markdown]
# 🔎 **Interpretación:** Ninguno de los tres ensambles supera a la regresión logística:
#
# | Ensamble | F1 en CV | Diferencia con la logística |
# |---|---|---|
# | Votación suave (logística + Random Forest + árbol) | 0,4643 | −0,005 |
# | XGBoost | 0,4606 | −0,009 |
# | Bagging de árboles | 0,4363 | −0,033 |
#
# Es un resultado **muy instructivo**:
# - La **votación** no mejora porque sus miembros cometen errores muy parecidos (usan la misma información) y el árbol débil resta.
# - El **boosting** reduce sesgo, pero aquí el límite no es el sesgo del algoritmo, sino la información que contienen las variables.
# - El **bagging** reduce la varianza del árbol (0,4298 → 0,4363), pero no alcanza a Random Forest, que además decorrelaciona los árboles.
#
# Los ensambles no son una garantía de mejora: ayudan cuando los modelos base se equivocan distinto o cuando el problema tiene alta varianza.

# %%
# 4.2.2 Ranking conjunto (clásicos + ensambles) por F1 en validación cruzada
df_todos = pd.concat([df_clasicos, df_ensambles]).sort_values("f1_mean", ascending=False).reset_index(drop=True)
todos_los_modelos = {**modelos_clasicos, **modelos_ensamble}   # Diccionario único con los 8 modelos sin entrenar

fig, eje = plt.subplots(figsize=(10, 4.5))
# xerr dibuja ± una desviación estándar: barras solapadas sugieren diferencias no concluyentes
eje.barh(df_todos["modelo"], df_todos["f1_mean"], xerr=df_todos["f1_std"], color="#4C72B0")
eje.invert_yaxis(); eje.set_xlabel("F1 medio en CV (± desviación)"); eje.set_title("Ranking de los 8 modelos")
plt.tight_layout(); guardar_figura("ranking_cv"); plt.show()
df_todos[["modelo", "f1_mean", "f1_std", "precision_mean", "recall_mean", "roc_auc_mean", "tiempo_s"]].round(4)

# %% [markdown]
# 🔎 **Interpretación:** El ranking conjunto de los 8 modelos, con barras de error, muestra un **grupo de cabeza** de cinco modelos entre 0,46 y 0,47 (logística, Random Forest, SVM, votación y XGBoost) que no se distinguen entre sí. Detrás quedan el bagging, el árbol y KNN.
# Los tres mejores (logística, Random Forest y SVM) pasan al ajuste de hiperparámetros. En la práctica, cuando varios modelos empatan conviene preferir el más **simple e interpretable**. Esa idea se retoma en la selección final.

# %% [markdown]
# ### 4.3 Ajuste de hiperparámetros

# %% [markdown]
# #### 📘 Concepto: Búsqueda de hiperparámetros
# Los **parámetros** se aprenden de los datos (coeficientes, divisiones de un árbol). Los **hiperparámetros** se fijan antes de
# entrenar y controlan la complejidad del modelo (`C`, `max_depth`, `learning_rate`…). Ajustarlos es un problema de
# optimización cuya función objetivo es el **F1 medio en validación cruzada**, nunca el test.
#
# | Estrategia | Cómo explora | Comentario |
# |---|---|---|
# | *Grid search* | Todas las combinaciones de una rejilla | Costo exponencial en el número de hiperparámetros |
# | *Random search* | $n$ combinaciones al azar de distribuciones | Con el mismo presupuesto suele encontrar mejores valores, porque pocos hiperparámetros importan de verdad (Bergstra y Bengio, 2012) |
# | *Successive halving* | Muchos candidatos con pocos datos; solo los mejores pasan a la siguiente ronda con más datos | Muy eficiente para modelos costosos como el SVM; con muy pocos datos en las primeras rondas la selección puede volverse ruidosa |
#
# **Distribuciones:** la log-uniforme se usa cuando importa el **orden de magnitud** (`C` de 0,001 a 100); la uniforme
# cuando importa el valor en un rango acotado (fracciones de 0,6 a 1,0).
#
# **Riesgo:** probar muchas combinaciones sobre los mismos folds sobreajusta a la validación (Cawley y Talbot, 2010). Por
# eso el test se reserva para la estimación final.

# %%
# 4.3.1 Espacios de búsqueda, qué controla cada hiperparámetro y por qué se eligió cada rango
# El prefijo "modelo__" dirige el hiperparámetro al paso "modelo" del pipeline (sintaxis paso__parametro)
ESPACIOS = {
    "Regresión Logística": {"modelo__C": loguniform(1e-3, 1e2)},
    "KNN": {"modelo__n_neighbors": randint(5, 101), "modelo__weights": ["uniform", "distance"], "modelo__p": [1, 2]},
    "Árbol de Decisión": {"modelo__max_depth": randint(3, 21), "modelo__min_samples_leaf": randint(5, 201),
                          "modelo__criterion": ["gini", "entropy"]},
    "SVM (RBF)": {"modelo__C": loguniform(1e-2, 1e2), "modelo__gamma": loguniform(1e-4, 1e0)},
    "Random Forest": {"modelo__n_estimators": randint(100, 501), "modelo__max_depth": [8, 12, 16, 20, None],
                      "modelo__min_samples_leaf": randint(1, 51), "modelo__max_features": ["sqrt", "log2", 0.5]},
    "Votación (soft)": {"modelo__weights": [[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [3, 2, 1], [1, 2, 3]]},
    "Bagging (árboles)": {"modelo__n_estimators": randint(50, 301), "modelo__max_samples": uniform(0.5, 0.5),
                          "modelo__max_features": uniform(0.5, 0.5), "modelo__estimator__max_depth": [None, 10, 15, 20]},
    "XGBoost": {"modelo__n_estimators": randint(100, 601), "modelo__learning_rate": loguniform(0.01, 0.3),
                "modelo__max_depth": randint(2, 9), "modelo__subsample": uniform(0.6, 0.4),
                "modelo__colsample_bytree": uniform(0.6, 0.4), "modelo__min_child_weight": randint(1, 11),
                "modelo__reg_lambda": loguniform(1e-2, 1e1)},
}
if isinstance(BALANCEADOR, SMOTE):   # El número de vecinos de SMOTE también es un hiperparámetro del pipeline
    for espacio in ESPACIOS.values():
        espacio["balanceo__k_neighbors"] = [3, 5, 7, 9]   # "balanceo__" apunta al paso de SMOTE

# Qué controla cada hiperparámetro (columna exigida por la rúbrica); las excepciones por modelo van aparte
QUE_CONTROLA = {
    "modelo__C": "Inverso de la regularización: valores altos ajustan más a los datos (menos sesgo, más varianza)",
    "modelo__n_neighbors": "Vecinos que votan: pocos = frontera irregular (varianza); muchos = frontera suave (sesgo)",
    "modelo__weights": "Peso de los vecinos: iguales ('uniform') o inversos a la distancia ('distance')",
    "modelo__p": "Métrica de Minkowski: 1 = Manhattan, 2 = Euclidiana",
    "modelo__max_depth": "Profundidad máxima de los árboles: limita las interacciones modeladas y el sobreajuste",
    "modelo__min_samples_leaf": "Mínimo de registros por hoja: suaviza las predicciones y reduce el sobreajuste",
    "modelo__criterion": "Medida de impureza para elegir divisiones (Gini o entropía)",
    "modelo__gamma": "Alcance del kernel RBF: alto = influencia local (fronteras complejas); bajo = fronteras suaves",
    "modelo__n_estimators": "Número de árboles del ensamble: más árboles reducen la varianza a mayor costo",
    "modelo__max_features": "Variables candidatas en cada división: decorrelaciona los árboles",
    "modelo__max_samples": "Fracción de registros (bootstrap) con la que se entrena cada árbol",
    "modelo__estimator__max_depth": "Profundidad de cada árbol base del bagging",
    "modelo__learning_rate": "Tamaño del paso con que cada árbol corrige a los anteriores (shrinkage)",
    "modelo__subsample": "Fracción de registros usada en cada ronda de boosting (regularización estocástica)",
    "modelo__colsample_bytree": "Fracción de variables usada por cada árbol",
    "modelo__min_child_weight": "Peso mínimo (hessiano) por hoja: valores altos hacen el modelo más conservador",
    "modelo__reg_lambda": "Regularización L2 sobre los pesos de las hojas",
    "balanceo__k_neighbors": "Vecinos que usa SMOTE para interpolar registros sintéticos de la clase minoritaria",
}
# Mismo nombre, distinto significado según el modelo: (modelo, hiperparámetro) -> descripción específica
QUE_CONTROLA_ESPECIFICO = {
    ("SVM (RBF)", "modelo__C"): "Penalización por violar el margen: alto = margen estrecho ajustado a los datos; bajo = margen amplio",
    ("Votación (soft)", "modelo__weights"): "Peso de cada modelo en el promedio de probabilidades del voto suave",
    ("Bagging (árboles)", "modelo__max_features"): "Fracción de variables que recibe cada árbol del bagging",
    ("XGBoost", "modelo__max_depth"): "Profundidad de cada árbol débil: orden de las interacciones capturadas",
    ("XGBoost", "modelo__n_estimators"): "Número de rondas de boosting (árboles secuenciales)",
}
# Por qué se eligió cada rango (columna exigida por la rúbrica)
RAZON_RANGO = {
    "modelo__C": "Escala logarítmica 10⁻³–10²: desde regularización fuerte hasta casi nula",
    "modelo__n_neighbors": "5–100: evita el ruido de k muy pequeño sin llegar a fronteras triviales",
    "modelo__weights": "Las dos opciones que ofrece scikit-learn",
    "modelo__p": "Las distancias más usadas con variables escaladas y binarias",
    "modelo__max_depth": "Desde árboles poco profundos hasta sin límite (None)",
    "modelo__min_samples_leaf": "Desde hojas finas hasta hojas grandes que suavizan",
    "modelo__criterion": "Los dos criterios clásicos de impureza",
    "modelo__gamma": "10⁻⁴–1 en escala log, alrededor del valor 'scale' ≈ 1/(n_variables·varianza)",
    "modelo__n_estimators": "Por encima del rango la ganancia suele ser marginal y crece el tamaño del modelo",
    "modelo__max_features": "'sqrt' y 'log2' (valores de la literatura) y 0,5 (más variables por división)",
    "modelo__max_samples": "50 %–100 % de los registros por estimador",
    "modelo__estimator__max_depth": "Árboles completos (None) o limitados para reducir varianza",
    "modelo__learning_rate": "0,01–0,3 en escala log: rango recomendado en la documentación de XGBoost",
    "modelo__subsample": "0,6–1,0: por debajo de 0,6 el modelo suele subajustar",
    "modelo__colsample_bytree": "0,6–1,0: por debajo de 0,6 se pierden variables relevantes",
    "modelo__min_child_weight": "1–10: de hojas finas a conservadoras",
    "modelo__reg_lambda": "0,01–10 en escala log alrededor del valor por defecto (1)",
    "balanceo__k_neighbors": "3–9 alrededor del valor por defecto de SMOTE (5)",
}
RAZON_RANGO_ESPECIFICO = {   # Razones que dependen del modelo
    ("Árbol de Decisión", "modelo__max_depth"): "3–20: por debajo de 3 subajusta; por encima de 20 memoriza 21 mil registros",
    ("Árbol de Decisión", "modelo__min_samples_leaf"): "5–200: hojas con entre ~0,03 % y ~1 % del entrenamiento",
    ("Random Forest", "modelo__min_samples_leaf"): "1–50: el promedio de árboles tolera hojas más finas",
    ("SVM (RBF)", "modelo__C"): "10⁻²–10² en escala log: rango estándar para SVM",
    ("Votación (soft)", "modelo__weights"): "Combinaciones que priorizan a cada modelo o los dejan iguales",
    ("XGBoost", "modelo__max_depth"): "2–8: los árboles débiles del boosting deben ser poco profundos",
}


def describir_rango(dist):
    """Texto legible de un espacio de búsqueda (lista o distribución de scipy)."""
    if isinstance(dist, list):                    # Opciones discretas: se listan
        return ", ".join(map(str, dist))
    a, b = dist.args[:2]                          # Parámetros con que se creó la distribución
    nombre = dist.dist.name                       # Tipo de distribución de scipy
    if nombre == "randint":                       # randint(a, b) genera enteros en [a, b-1]
        return f"enteros en [{a}, {b - 1}]"
    if nombre in ("loguniform", "reciprocal"):    # Uniforme en escala logarítmica
        return f"log-uniforme en [{a:g}, {b:g}]"
    if nombre == "uniform":                       # uniform(loc, scale) cubre [loc, loc + scale]
        return f"uniforme en [{a:g}, {a + b:g}]"
    return str(dist)


# Vista de todos los espacios definidos (solo se buscará en los 3 mejores modelos)
pd.DataFrame([{"modelo": m, "hiperparametro": p, "rango": describir_rango(d)} for m, e in ESPACIOS.items() for p, d in e.items()])

# %% [markdown]
# 🔎 **Interpretación:** La tabla lista los espacios de búsqueda de los 8 modelos, aunque solo se ajustarán los 3 mejores del ranking.
# Observa las distribuciones:
# - **Log-uniforme:** para `C` y `gamma`, donde importa el orden de magnitud.
# - **Enteros:** para `n_estimators` y `max_depth`.
# - **Listas:** para opciones discretas.
#
# No aparece `k_neighbors` de SMOTE porque el balanceador elegido fue el submuestreo, que no tiene ese hiperparámetro. El código lo agrega solo si corresponde.

# %%
# 4.3.2 Búsqueda aleatoria (o por mitades si el SVM se entrena con SMOTE) sobre los 3 mejores modelos del ranking
MODELOS_A_AJUSTAR = df_todos["modelo"].head(3).tolist()   # Presupuesto de cómputo concentrado en los candidatos fuertes
busquedas = {}
for nombre in MODELOS_A_AJUSTAR:
    inicio = time.time()
    pipe = crear_pipeline(clone(todos_los_modelos[nombre]))   # Pipeline completo: sin fuga durante la búsqueda
    # Mitades sucesivas solo si el SVM es costoso (con SMOTE el entrenamiento crece a ~28 mil filas por fold).
    # Con submuestreo cada fold queda en ~6 mil filas y la búsqueda aleatoria es barata. Además, en una prueba
    # previa las primeras rondas por mitades usaban tan pocos datos que el F1 colapsaba y la selección era casi al azar.
    if nombre == "SVM (RBF)" and isinstance(BALANCEADOR, SMOTE):
        # Mitades sucesivas: 16 candidatos con pocos datos; en cada ronda sobrevive la mitad y recibe el doble de datos
        busqueda = HalvingRandomSearchCV(pipe, ESPACIOS[nombre], n_candidates=16, factor=2, cv=CV, scoring="f1",
                                         random_state=RANDOM_STATE, n_jobs=-1)
    else:
        # return_train_score permite comparar F1 de entrenamiento y de validación (diagnóstico de sobreajuste)
        busqueda = RandomizedSearchCV(pipe, ESPACIOS[nombre], n_iter=N_ITER_BUSQUEDA, cv=CV, scoring="f1",
                                      random_state=RANDOM_STATE, n_jobs=-1, return_train_score=True)
    busquedas[nombre] = busqueda.fit(X_train, y_train)   # Al terminar, reentrena la mejor combinación con todo el 70 %
    print(f"{nombre:<22} mejor F1 CV = {busqueda.best_score_:.4f} ({time.time() - inicio:.0f} s)\n  {busqueda.best_params_}")

# %% [markdown]
# 🔎 **Interpretación:** Resultados de la búsqueda aleatoria (20 combinaciones × 5 folds por modelo):
# - **Regresión Logística:** mejor `C = 0,144` (algo más de regularización que el valor por defecto, 1,0). F1 en CV = 0,4701, en unos 30 s.
# - **Random Forest:** 341 árboles, `max_depth = 20`, `min_samples_leaf = 14`, `max_features = 'sqrt'`. F1 = 0,4696, en 2–4 minutos.
# - **SVM:** `C = 0,67` y `gamma = 0,138`. F1 = 0,4655, en unos 15–20 minutos: fue, con diferencia, la búsqueda más costosa. (Los tiempos exactos aparecen en la salida y varían entre ejecuciones; el orden de magnitud es estable.)
#
# Las mejoras frente a los valores por defecto son mínimas (≤ 0,0006). Los hiperparámetros por defecto ya estaban cerca del óptimo y **el límite está en los datos, no en la configuración**.
#
# *Nota metodológica:* una prueba previa con búsqueda por mitades sucesivas para el SVM colapsó (F1 ≈ 0,37), porque sus primeras rondas usaban tan pocos datos que el F1 no discriminaba. Por eso se usa búsqueda aleatoria cuando el balanceo es por submuestreo.

# %%
# 4.3.3 Tabla explicativa: qué controla, rango, razón, valor final y efecto observado de cada hiperparámetro
def resultados_busqueda(busqueda):
    """cv_results_ como DataFrame; en búsqueda por mitades se usa la 1.ª ronda (todos con el mismo recurso)."""
    res = pd.DataFrame(busqueda.cv_results_)
    return res[res["iter"] == 0] if "iter" in res else res   # Comparar candidatos con distinto recurso sería injusto


def es_numerico(valores):
    """True si todos los valores probados son números (no listas, textos ni None)."""
    return all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) for v in valores)


def efecto_hiperparametro(busqueda, param):
    """Resume cómo se asoció el valor del hiperparámetro con el F1 medio de validación."""
    res = resultados_busqueda(busqueda)
    valores, score = list(res[f"param_{param}"]), res["mean_test_score"].to_numpy()
    if es_numerico(valores):
        # Spearman: ¿a mayor valor del hiperparámetro, mayor (o menor) F1? Es una asociación, no un efecto aislado
        rho, p_valor = stats.spearmanr(np.array(valores, dtype=float), score)
        if p_valor < 0.05:
            return f"ρ Spearman = {rho:+.2f} (p = {p_valor:.3f}): aumentarlo {'mejoró' if rho > 0 else 'empeoró'} el F1"
        return f"ρ Spearman = {rho:+.2f} (p = {p_valor:.3f}): sin efecto significativo en el rango"
    # Categórico: F1 medio de las combinaciones que usaron cada opción
    etiquetas = [etiqueta_valor(v) for v in valores]   # Etiquetas legibles (una Serie convertiría 16 en 16.0 y None en nan)
    medias = pd.Series(score).groupby(etiquetas).mean().sort_values(ascending=False)
    return "F1 medio por valor: " + "; ".join(f"{k} = {v:.3f}" for k, v in medias.items())


def etiqueta_valor(valor):
    """Texto legible de un valor probado: 'None' en lugar de nan y enteros sin decimales."""
    if valor is None or (isinstance(valor, (float, np.floating)) and np.isnan(valor)):
        return "None"
    if isinstance(valor, (float, np.floating)) and float(valor).is_integer():
        return str(int(valor))   # 16.0 -> "16"
    return str(valor)


def formatear(valor):
    """Cuatro cifras significativas para decimales; texto para el resto."""
    return f"{valor:.4g}" if isinstance(valor, (float, np.floating)) else str(valor)


filas = []
for nombre, busqueda in busquedas.items():
    for param, dist in ESPACIOS[nombre].items():   # Una fila por hiperparámetro ajustado (tabla 4.3 del informe)
        filas.append({"modelo": nombre,
                      "hiperparametro": param.replace("modelo__", "").replace("balanceo__", "SMOTE: "),
                      "que_controla": QUE_CONTROLA_ESPECIFICO.get((nombre, param), QUE_CONTROLA[param]),
                      "rango_probado": describir_rango(dist),
                      "razon_rango": RAZON_RANGO_ESPECIFICO.get((nombre, param), RAZON_RANGO[param]),
                      "valor_final": formatear(busqueda.best_params_[param]),
                      "efecto": efecto_hiperparametro(busqueda, param)})
tabla_hiper = guardar_tabla(pd.DataFrame(filas), "ajuste_hiperparametros")
tabla_hiper

# %% [markdown]
# 🔎 **Interpretación:** La tabla explicativa (tabla 4.3 del informe) resume por hiperparámetro qué controla, el rango, la razón del rango, el valor final y el efecto observado:
# - **`C` de la logística:** aumentarlo mejoró el F1 (ρ = +0,63, p = 0,003). Los valores muy pequeños (regularización excesiva) empeoran el modelo y, a partir de ~0,1, el F1 se estabiliza.
# - **`C` del SVM:** también tiene efecto positivo significativo (ρ = +0,69).
# - **Random Forest:** ninguno de sus hiperparámetros tiene efecto significativo en los rangos probados (p > 0,4). Es un modelo robusto a su configuración, como anticipa el 📘.
# - **`gamma` del SVM:** su efecto no es significativo en el rango (p = 0,125).

# %%
# 4.3.4 Efecto visual de cada hiperparámetro sobre el F1 de validación (un gráfico por modelo ajustado)
for nombre, busqueda in busquedas.items():
    res = resultados_busqueda(busqueda)
    params = list(ESPACIOS[nombre])
    fig, ejes = plt.subplots(1, len(params), figsize=(3.8 * len(params), 3.4), squeeze=False)   # Un panel por hiperparámetro
    for eje, param in zip(ejes[0], params):
        valores = list(res[f"param_{param}"])
        if es_numerico(valores):
            eje.scatter(np.array(valores, dtype=float), res["mean_test_score"], alpha=0.7)   # Nube: tendencia del F1
        else:
            etiquetas = pd.Series(valores).astype(str)
            medias = res["mean_test_score"].groupby(etiquetas.to_numpy()).mean()   # F1 medio por opción
            eje.bar(medias.index, medias.values, color="#4C72B0"); eje.tick_params(axis="x", rotation=45)
        eje.set_title(param.replace("modelo__", "").replace("balanceo__", "SMOTE "), fontsize=9)
        eje.set_ylabel("F1 CV")
    fig.suptitle(f"{nombre}: F1 de validación según cada hiperparámetro")
    plt.tight_layout(); guardar_figura(f"hiperparametros_{ALIAS.get(nombre, nombre.split()[0].lower())}"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** Los gráficos muestran la forma del efecto:
# - **Logística:** el F1 sube rápidamente cuando `C` sale de valores cercanos a 0 y luego forma una **meseta** (~0,469–0,470). Más allá de C ≈ 0,1 la regularización deja de importar.
# - **Random Forest:** nubes sin tendencia, coherentes con la ausencia de efecto significativo.
# - **SVM:** los valores de `C` muy pequeños dan los peores F1.
#
# Leer estas gráficas ayuda a decidir si vale la pena ampliar un rango. Aquí ninguna curva sugiere que el óptimo esté fuera de los rangos probados.

# %% [markdown]
# #### 📘 Concepto: Umbral de decisión
# Un clasificador probabilístico entrega $\hat p(x)$ y decide $\hat y = \mathbb{1}[\hat p(x) \ge t]$. El umbral por defecto
# $t = 0{,}5$ no es óptimo cuando las clases están desbalanceadas o cuando los errores tienen costos distintos:
# - Bajar $t$ ⇒ más positivos predichos ⇒ **sube el Recall** y baja la Precision.
# - Subir $t$ ⇒ lo contrario.
#
# `TunedThresholdClassifierCV` busca el $t$ que maximiza el F1 **con validación cruzada sobre el entrenamiento**, de modo que
# el test no se usa para elegirlo. El umbral es, en la práctica, un hiperparámetro más del sistema de decisión.

# %%
# 4.3.5 Ajuste del umbral de decisión (maximiza F1 en CV usando solo entrenamiento) y mejora frente a la versión base
modelos_umbral = {}
for nombre, busqueda in busquedas.items():
    # Se parte del mejor pipeline de la búsqueda; el umbral se elige con los mismos 5 folds
    modelos_umbral[nombre] = TunedThresholdClassifierCV(busqueda.best_estimator_, scoring="f1", cv=CV,
                                                        n_jobs=-1, random_state=RANDOM_STATE).fit(X_train, y_train)
# Evolución del F1 en CV: base (4.1/4.2) -> hiperparámetros ajustados -> + umbral óptimo
tabla_mejora = pd.DataFrame([{
    "modelo": nombre,
    "f1_cv_base": df_todos.set_index("modelo").loc[nombre, "f1_mean"],
    "f1_cv_ajustado": busqueda.best_score_,
    "f1_cv_ajustado_umbral": modelos_umbral[nombre].best_score_,
    "umbral_optimo": modelos_umbral[nombre].best_threshold_} for nombre, busqueda in busquedas.items()])
tabla_mejora["mejora_total"] = tabla_mejora["f1_cv_ajustado_umbral"] - tabla_mejora["f1_cv_base"]   # Ganancia acumulada
guardar_tabla(tabla_mejora.round(4), "mejora_ajuste")
tabla_mejora.round(4)

# %% [markdown]
# 🔎 **Interpretación:** El ajuste del **umbral de decisión** aporta la mayor mejora de toda la fase:
#
# | Modelo | F1 en CV | Umbral óptimo |
# |---|---|---|
# | Regresión Logística | 0,4701 → 0,4856 | 0,615 |
# | Random Forest | 0,4696 → 0,4815 | 0,580 |
# | SVM | 0,4655 → 0,4748 | 0,432 |
#
# **Por qué:** el submuestreo deja al modelo "creyendo" que la mitad de los estudiantes son positivos, así que sus probabilidades quedan infladas. Subir el umbral por encima de 0,5 corrige ese sesgo y equilibra Precision y Recall. Es la idea del 📘: el umbral es un hiperparámetro más, y en problemas desbalanceados suele importar más que la configuración del algoritmo.

# %% [markdown]
# ### 🧠 Para profundizar — Fase 4: Modelamiento
# **Ideas clave**
# - La validación cruzada estima el desempeño esperado **y su variabilidad**; comparar solo medias puede engañar.
# - Todo lo que aprende de los datos va en el pipeline, para que la validación sea honesta y el despliegue sea idéntico.
# - Bagging reduce varianza, boosting reduce sesgo y la votación aprovecha modelos que se equivocan distinto.
# - Los hiperparámetros se eligen en validación; el umbral de decisión también es un hiperparámetro.
#
# **Preguntas de repaso**
# 1. ¿Por qué Random Forest suele superar a un árbol de decisión individual? Explícalo con la fórmula de varianza del promedio.
# 2. ¿Qué diferencia hay entre `max_depth` en Random Forest y en XGBoost, y por qué los rangos elegidos son distintos?
# 3. ¿Por qué la búsqueda aleatoria es más eficiente que la búsqueda en rejilla con el mismo número de pruebas?
# 4. Si el F1 de entrenamiento es 0,95 y el de validación 0,50, ¿qué está pasando y qué hiperparámetros lo corregirían?
#
# **Ejercicios**
# - Agrega `LightGBM` como segundo modelo de boosting y compáralo con XGBoost en tiempo y F1.
# - Duplica `N_ITER_BUSQUEDA` y verifica si el mejor F1 en CV cambia de forma relevante (rendimientos decrecientes).
#
# **Referencias**
# - James, G., Witten, D., Hastie, T. y Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2.ª ed.). Springer (caps. 4, 5, 8 y 9).
# - Hastie, T., Tibshirani, R. y Friedman, J. (2009). *The Elements of Statistical Learning* (2.ª ed.). Springer (caps. 7, 10, 12 y 15).
# - Breiman, L. (1996). Bagging predictors. *Machine Learning*, 24, 123–140. — Breiman, L. (2001). Random forests. *Machine Learning*, 45, 5–32.
# - Cortes, C. y Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20, 273–297.
# - Friedman, J. H. (2001). Greedy function approximation: a gradient boosting machine. *Annals of Statistics*, 29(5), 1189–1232.
# - Chen, T. y Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *KDD '16*, 785–794.
# - Bergstra, J. y Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR*, 13, 281–305.
# - Cawley, G. C. y Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent selection bias in performance evaluation. *JMLR*, 11, 2079–2107.
