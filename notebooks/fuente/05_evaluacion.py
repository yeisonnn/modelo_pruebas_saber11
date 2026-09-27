# %% [markdown]
# ## 5. Evaluación
# ### 5.1 Resultados en el conjunto de prueba (30 %)

# %% [markdown]
# #### 📘 Concepto: Métricas de clasificación
# Todas parten de la **matriz de confusión**:
#
# | | Predicho 1 | Predicho 0 |
# |---|---|---|
# | **Real 1** | VP (verdadero positivo) | FN (falso negativo) |
# | **Real 0** | FP (falso positivo) | VN (verdadero negativo) |
#
# | Métrica | Fórmula | Pregunta que responde |
# |---|---|---|
# | Accuracy | $\frac{VP+VN}{VP+VN+FP+FN}$ | ¿Qué fracción acierta? (engañosa con desbalance) |
# | Precision | $\frac{VP}{VP+FP}$ | De los que predigo "alto", ¿cuántos lo son? |
# | Recall (sensibilidad) | $\frac{VP}{VP+FN}$ | De los que son "alto", ¿cuántos detecto? |
# | F1 | $2\cdot\frac{P\cdot R}{P+R}$ | Media armónica: solo es alta si P **y** R lo son |
# | ROC-AUC | Área bajo la curva TPR vs FPR | Probabilidad de que un positivo reciba mayor puntaje que un negativo (independiente del umbral) |
# | PR-AUC | Área bajo la curva Precision-Recall | Más informativa que ROC-AUC con clases desbalanceadas (Saito y Rehmsmeier, 2015); el azar vale la prevalencia (~0,17) |
#
# El conjunto de prueba se usa **una sola vez**, al final, para estimar el desempeño en datos no vistos.

# %%
# 5.1.1 Utilidades de evaluación y evaluación de todos los candidatos en test
def puntaje_positivo(modelo, X):
    """Probabilidad de la clase positiva (o función de decisión si el modelo no expone probabilidades)."""
    return modelo.predict_proba(X)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X)


def metricas_test(nombre, modelo, X, y):
    """Métricas de umbral (con la clase predicha) y de ranking (con el puntaje continuo)."""
    prediccion, puntaje = modelo.predict(X), puntaje_positivo(modelo, X)
    return {"modelo": nombre, "accuracy": accuracy_score(y, prediccion), "precision": precision_score(y, prediccion),
            "recall": recall_score(y, prediccion), "f1": f1_score(y, prediccion),
            "roc_auc": roc_auc_score(y, puntaje), "pr_auc": average_precision_score(y, puntaje)}


# Los 8 modelos base se reentrenan con TODO el 70 %; los ajustados ya fueron reentrenados por la búsqueda
candidatos = {}
for nombre, modelo in todos_los_modelos.items():
    candidatos[f"{nombre} (base)"] = crear_pipeline(clone(modelo)).fit(X_train, y_train)
for nombre, busqueda in busquedas.items():
    candidatos[f"{nombre} (ajustado)"] = busqueda.best_estimator_                  # Umbral por defecto (0,5)
    candidatos[f"{nombre} (ajustado + umbral)"] = modelos_umbral[nombre]          # Umbral óptimo de 4.3.5

# Primera y única evaluación en test: 8 base + 3 ajustados + 3 ajustados con umbral = 14 candidatos
df_test = pd.DataFrame([metricas_test(n, m, X_test, y_test) for n, m in candidatos.items()])
df_test = df_test.sort_values("f1", ascending=False).reset_index(drop=True)
guardar_tabla(df_test.round(4), "resultados_test")   # Tabla 5.1 del informe
df_test.round(4)

# %% [markdown]
# 🔎 **Interpretación:** En el test (9.000 estudiantes nunca usados), los tres modelos con umbral ajustado quedan arriba:
#
# | Modelo | F1 | Precision | Recall | Accuracy |
# |---|---|---|---|---|
# | Regresión Logística + umbral | 0,4730 | 0,400 | 0,580 | 0,779 |
# | Random Forest + umbral | 0,4726 | — | — | — |
# | SVM + umbral | 0,4707 | — | — | — |
#
# - Sin ajustar el umbral, el Recall es mayor (~0,74) pero la Precision cae a ~0,33. El umbral cambia el **punto de operación**, no la capacidad de ordenar: la ROC-AUC de la logística es 0,7975 con y sin umbral.
# - Los resultados en test son muy parecidos a los de CV (0,47 frente a 0,486), así que la validación cruzada fue una buena estimación.

# %%
# 5.1.2 Comparación visual del F1 en test de todos los candidatos
# Colores distintos para los ajustados: permite ver el aporte del ajuste de hiperparámetros y del umbral
fig, eje = plt.subplots(figsize=(10, 6))
colores = ["crimson" if "ajustado" in n else "#4C72B0" for n in df_test["modelo"]]
eje.barh(df_test["modelo"], df_test["f1"], color=colores)
eje.invert_yaxis(); eje.set_xlabel("F1 en test"); eje.set_title("F1 en el conjunto de prueba (rojo = ajustados)")
plt.tight_layout(); guardar_figura("f1_test"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** El gráfico separa dos grupos: arriba, los modelos con umbral ajustado (F1 ≈ 0,47); en el medio, los modelos base y ajustados sin umbral de la familia fuerte, XGBoost y la votación (0,45–0,46); abajo, el bagging, el árbol y KNN (0,42–0,43).
# El salto más grande no viene de cambiar de algoritmo, sino de **ajustar el umbral**. Esta lección se repite en muchos problemas desbalanceados reales.

# %% [markdown]
# ### 5.2 Selección del modelo final
# El modelo se elige por su **F1 en validación cruzada**, no por el de test: elegir con el test introduce sesgo de selección
# (el test dejaría de ser una estimación independiente). El test se usa para confirmar y para las pruebas estadísticas.

# %%
# 5.2.1 Candidato final (mayor F1 en CV) y rival más fuerte de otra familia de modelos
cv_f1 = {}   # F1 en CV de cada candidato ajustado (con y sin umbral óptimo)
for nombre, busqueda in busquedas.items():
    cv_f1[f"{nombre} (ajustado)"] = busqueda.best_score_
    cv_f1[f"{nombre} (ajustado + umbral)"] = modelos_umbral[nombre].best_score_
NOMBRE_FINAL = max(cv_f1, key=cv_f1.get)                   # Criterio de selección: F1 en validación
FAMILIA_FINAL = NOMBRE_FINAL.split(" (")[0]                # Nombre del algoritmo sin el sufijo
# El rival es el mejor candidato de OTRO algoritmo: compararlo con su propia variante no aportaría evidencia
RIVAL = max((k for k in cv_f1 if k.split(" (")[0] != FAMILIA_FINAL), key=cv_f1.get)
FAMILIA_RIVAL = RIVAL.split(" (")[0]
modelo_final = candidatos[NOMBRE_FINAL]

# Comparación lado a lado del F1 en CV y en test: si el orden se mantiene, la selección es consistente
tabla_seleccion_cv = (pd.DataFrame({"candidato": list(cv_f1), "f1_cv": list(cv_f1.values())})
                      .merge(df_test[["modelo", "f1"]].rename(columns={"modelo": "candidato", "f1": "f1_test"}), on="candidato")
                      .sort_values("f1_cv", ascending=False).round(4))
print(f"Modelo final: {NOMBRE_FINAL} | Rival: {RIVAL}")
tabla_seleccion_cv

# %% [markdown]
# 🔎 **Interpretación:** El candidato con mayor F1 en validación cruzada es la **Regresión Logística (ajustado + umbral)** (0,4856). El rival más fuerte de otra familia es Random Forest (ajustado + umbral), con 0,4815.
# El orden por CV coincide con el orden en test (0,4730 frente a 0,4726): la selección es **consistente**. Elegir por CV y confirmar con el test evita el sesgo de selección descrito en 5.2.

# %% [markdown]
# #### 📘 Concepto: Comparación estadística de modelos
# Que un modelo tenga un F1 mayor no prueba que sea **mejor**: la diferencia puede deberse al azar de la muestra. Se usan
# tres herramientas complementarias:
#
# 1. **Intervalo de confianza bootstrap** (Efron y Tibshirani, 1993): se remuestrea el test con reemplazo $B = 1000$ veces
#    y se calcula el F1 en cada remuestra. Los percentiles 2,5 y 97,5 dan un IC del 95 %. Si los IC de dos modelos se
#    solapan mucho, la diferencia es incierta.
# 2. **Prueba de McNemar** (Dietterich, 1998): compara dos modelos **en los mismos registros de test**. Solo cuentan los
#    desacuerdos: $b$ = casos donde acierta A y falla B, $c$ = el caso contrario. Bajo $H_0$ (igual tasa de error),
#    $b \sim \text{Binomial}(b+c,\ 0{,}5)$.
# 3. **t-test corregido de Nadeau y Bengio (2003):** compara los F1 de los mismos $k$ folds. Los folds comparten datos de
#    entrenamiento, así que el t-test clásico subestima la varianza. La corrección es:
#    $$t = \frac{\bar d}{\sqrt{\left(\frac{1}{k} + \frac{n_{test}}{n_{train}}\right)\hat\sigma_d^2}}$$
#    con $d_j$ = diferencia de F1 en el fold $j$ y $\frac{n_{test}}{n_{train}} = \frac{1}{k-1}$ en k-fold.
#
# Un p-valor < 0,05 indica evidencia de diferencia real. Si no la hay, conviene preferir el modelo más **simple o interpretable**.

# %%
# 5.2.2 Justificación estadística: IC bootstrap del F1, McNemar en test y t-test corregido (Nadeau-Bengio) en CV
def ic_bootstrap_f1(y_real, y_pred, n_boot=1000):
    """Intervalo de confianza del 95 % del F1 remuestreando el conjunto de prueba."""
    rng = np.random.default_rng(RANDOM_STATE)                    # Generador reproducible
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    # Cada remuestra toma n índices con reemplazo (algunos registros se repiten y otros quedan fuera)
    valores = [f1_score(y_real[i], y_pred[i]) for i in (rng.integers(0, len(y_real), len(y_real)) for _ in range(n_boot))]
    return np.percentile(valores, [2.5, 97.5])                   # Percentiles = límites del IC 95 %


def mcnemar_exacto(y_real, pred_a, pred_b):
    """Prueba exacta de McNemar: ¿los dos modelos se equivocan en proporciones distintas?"""
    y_real, pred_a, pred_b = map(np.asarray, (y_real, pred_a, pred_b))
    b = int(np.sum((pred_a == y_real) & (pred_b != y_real)))   # A acierta, B falla
    c = int(np.sum((pred_a != y_real) & (pred_b == y_real)))   # A falla, B acierta
    # Prueba binomial bilateral exacta sobre los desacuerdos (válida incluso con pocos desacuerdos)
    return b, c, (stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c > 0 else 1.0)


def ttest_corregido(a, b):
    """t-test pareado con varianza corregida por solapamiento de folds (Nadeau y Bengio, 2003)."""
    d = np.asarray(a) - np.asarray(b)                        # Diferencia de F1 fold a fold
    k = len(d)
    var_corregida = (1 / k + 1 / (k - 1)) * d.var(ddof=1)   # n_test/n_train = 1/(k-1) en k-fold
    t = d.mean() / np.sqrt(var_corregida)
    return t, 2 * stats.t.sf(abs(t), df=k - 1)               # p-valor bilateral con k-1 grados de libertad


def f1_folds(busqueda):
    """F1 de cada fold para la mejor configuración encontrada."""
    return np.array([busqueda.cv_results_[f"split{k}_test_score"][busqueda.best_index_] for k in range(CV.get_n_splits())])


predicciones = {n: candidatos[n].predict(X_test) for n in cv_f1}   # Predicciones de test de los candidatos ajustados
filas = []
for n in cv_f1:
    inferior, superior = ic_bootstrap_f1(y_test, predicciones[n])
    filas.append({"candidato": n, "f1_test": f1_score(y_test, predicciones[n]), "ic95_inf": inferior, "ic95_sup": superior})
df_ic = pd.DataFrame(filas).sort_values("f1_test", ascending=False).round(4)

# Final vs rival en test (McNemar) y en CV (t corregido); además, final vs la línea base de la fase 1
b_mc, c_mc, p_mcnemar = mcnemar_exacto(y_test, predicciones[NOMBRE_FINAL], predicciones[RIVAL])
_, p_t_rival = ttest_corregido(f1_folds(busquedas[FAMILIA_FINAL]), f1_folds(busquedas[FAMILIA_RIVAL]))
_, p_t_base = ttest_corregido(f1_folds(busquedas[FAMILIA_FINAL]), f1_por_fold["Regresión Logística"])
print(f"McNemar {NOMBRE_FINAL} vs {RIVAL}: b = {b_mc}, c = {c_mc}, p = {p_mcnemar:.4g}")
print(f"t-test corregido en CV vs {FAMILIA_RIVAL}: p = {p_t_rival:.4g} | vs Regresión Logística base: p = {p_t_base:.4g}")
df_ic

# %% [markdown]
# 🔎 **Interpretación:** Tres pruebas complementarias:
# - **IC bootstrap al 95 %:** logística 0,453–0,494; Random Forest 0,452–0,492; SVM 0,451–0,491. Los intervalos se **solapan casi por completo**: en F1 los tres modelos son estadísticamente indistinguibles.
# - **McNemar (logística frente a Random Forest):** de los 669 casos en que discrepan, la logística acierta 364 y Random Forest 305 (p = 0,025). La logística comete **significativamente menos errores totales** (su accuracy es mayor: 0,779 frente a 0,772), aunque su F1 sea igual.
# - **t-test corregido de Nadeau–Bengio en CV:** frente a Random Forest, p = 0,895; frente a la logística base, p = 0,670. No hay diferencia real de F1, y ajustar `C` no cambió el F1 de forma significativa: la ganancia vino del umbral.
#
# **Conclusión:** con F1 equivalente, menos errores totales, un modelo de 20 KB y coeficientes interpretables, la **regresión logística es la mejor elección**. Es un ejemplo del principio de parsimonia.

# %%
# 5.2.3 Matriz de confusión del modelo final (se guarda también para mostrarla en la app)
fig, eje = plt.subplots(figsize=(5.5, 4.5))
# values_format="d": conteos enteros; la diagonal son aciertos (VN, VP) y fuera de ella los errores (FP, FN)
ConfusionMatrixDisplay.from_predictions(y_test, predicciones[NOMBRE_FINAL], display_labels=["No alto", "Alto"],
                                        cmap="Blues", values_format="d", ax=eje)
eje.set_title(f"Matriz de confusión — {NOMBRE_FINAL}", fontsize=10)
plt.tight_layout(); guardar_figura("matriz_confusion_final"); plt.show()
# classification_report: precision, recall y F1 de cada clase, más sus promedios
print(classification_report(y_test, predicciones[NOMBRE_FINAL], target_names=["No alto", "Alto"], digits=4))

# %% [markdown]
# 🔎 **Interpretación:** La matriz de confusión del modelo final en test:
#
# | | Predicho alto | Predicho no alto |
# |---|---|---|
# | **Real alto** (1.541) | 893 detectados (VP) | 648 no detectados (FN) |
# | **Real no alto** (7.459) | 1.342 falsos positivos (FP) | 6.117 correctos (VN) |
#
# - De cada 10 estudiantes señalados como "alto", 4 lo son (Precision 0,40). Se detecta al 58 % de los que realmente lo son (Recall 0,58).
# - Para la clase "no alto" el desempeño es alto (F1 0,86): es la clase fácil y mayoritaria.
#
# En un uso de focalización, los falsos positivos son estudiantes con buen perfil socioeconómico que no alcanzaron 300. Los falsos negativos son estudiantes que superaron las expectativas de su contexto. Ambos grupos muestran los límites de predecir con variables estructurales.

# %%
# 5.2.4 Curvas ROC y Precisión-Recall de los modelos ajustados (mismo puntaje con o sin umbral)
# Las curvas recorren todos los umbrales posibles: comparan la capacidad de ORDENAR, no una decisión concreta
fig, ejes = plt.subplots(1, 2, figsize=(14, 5))
for nombre, busqueda in busquedas.items():
    puntaje = puntaje_positivo(busqueda.best_estimator_, X_test)
    RocCurveDisplay.from_predictions(y_test, puntaje, name=nombre, ax=ejes[0])
    PrecisionRecallDisplay.from_predictions(y_test, puntaje, name=nombre, ax=ejes[1])
ejes[0].plot([0, 1], [0, 1], "k--", lw=1); ejes[0].set_title("Curvas ROC (test)")   # Diagonal = clasificador aleatorio
ejes[1].axhline(y_test.mean(), ls="--", color="k", lw=1, label="Azar")              # En PR el azar es la prevalencia
ejes[1].set_title("Curvas Precisión-Recall (test)")
plt.tight_layout(); guardar_figura("roc_pr"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** - **Curvas ROC:** la logística y Random Forest son casi idénticas (AUC ≈ 0,80) y el SVM queda algo por debajo (0,78).
# - **Curvas Precisión-Recall:** Random Forest tiene la mejor PR-AUC (0,48), seguida de la logística (0,46). El SVM queda en 0,40, porque sus puntajes de decisión ordenan peor a los casos más seguros. Todas superan con amplitud la línea del azar (prevalencia 0,17).
#
# Random Forest es algo mejor en la zona de **alta precisión y bajo recall**: si el objetivo fuera identificar solo a los estudiantes con probabilidad muy alta, podría convenir. En el punto de operación elegido (F1 máximo), ambos modelos coinciden.

# %% [markdown]
# #### 📘 Concepto: Curva de aprendizaje
# Grafica el desempeño en entrenamiento y en validación a medida que crece el número de registros de entrenamiento:
# - **Curvas que convergen en un valor bajo:** alto **sesgo**. Más datos no ayudan; hacen falta mejores variables o un modelo más flexible.
# - **Brecha grande entre ambas curvas:** alta **varianza**. Más datos o más regularización ayudan.
# - **Curva de validación que se aplana (meseta):** agregar datos casi no mejora el modelo.
#
# Aquí justifica empíricamente haber modelado con una muestra de 30.000 registros en lugar de los ~72 mil registros limpios disponibles.

# %%
# 5.2.5 Curva de aprendizaje del modelo final: ¿más datos mejorarían el F1? (justifica la muestra de 30.000)
# train_sizes: 6 tamaños entre el 10 % y el 100 % del entrenamiento; en cada uno se hace la CV de 5 folds
tamanos, f1_entrena, f1_valida = learning_curve(busquedas[FAMILIA_FINAL].best_estimator_, X_train, y_train, cv=CV,
                                                 scoring="f1", train_sizes=np.linspace(0.1, 1.0, 6), n_jobs=-1)
fig, eje = plt.subplots(figsize=(8, 4))
eje.plot(tamanos, f1_entrena.mean(axis=1), marker="o", label="Entrenamiento")   # Media de los 5 folds
eje.plot(tamanos, f1_valida.mean(axis=1), marker="o", label="Validación")
# Banda de ± 1 desviación: variabilidad entre folds
eje.fill_between(tamanos, f1_valida.mean(axis=1) - f1_valida.std(axis=1), f1_valida.mean(axis=1) + f1_valida.std(axis=1), alpha=0.2)
eje.set_xlabel("Registros de entrenamiento"); eje.set_ylabel("F1"); eje.set_title(f"Curva de aprendizaje — {FAMILIA_FINAL}"); eje.legend()
plt.tight_layout(); guardar_figura("curva_aprendizaje"); plt.show()

# %% [markdown]
# 🔎 **Interpretación:** La curva de aprendizaje de la regresión logística muestra que el F1 de entrenamiento y el de validación **convergen alrededor de 0,47** a partir de unos 10 mil registros. Con 17 mil la brecha es prácticamente nula.
# Según el 📘, este patrón indica **alto sesgo**, no alta varianza: agregar más registros (por ejemplo, los 72 mil disponibles) apenas mejoraría el modelo. Esto **justifica empíricamente** haber modelado con 30 mil registros. Para mejorar habría que aportar **variables más informativas**, como el efecto colegio (extensión E1).

# %% [markdown]
# #### 📘 Concepto: Importancia por permutación
# Mide cuánto **empeora** la métrica cuando se desordenan al azar los valores de una variable en el test. Así se rompe su
# relación con el objetivo sin cambiar su distribución:
# $$I_j = s(\text{modelo}, X) - \frac{1}{R}\sum_{r=1}^{R} s\big(\text{modelo}, X^{(j,\,r)}_{\text{permutada}}\big)$$
# - **Ventajas:** sirve con cualquier modelo (incluidos ensambles y pipelines), se mide en la métrica de negocio (F1) y
#   actúa sobre las variables **originales**, no sobre las columnas codificadas.
# - **Limitaciones:** con variables correlacionadas reparte (o subestima) la importancia, y describe asociación **en el
#   modelo**, no causalidad en el mundo (Molnar, 2022).

# %%
# 5.2.6 Importancia de variables por permutación en test (caída del F1 al desordenar cada variable)
# n_repeats=5: cada variable se permuta 5 veces y se promedia, para reducir el ruido del azar
importancia = permutation_importance(modelo_final, X_test, y_test, scoring="f1", n_repeats=5,
                                     random_state=RANDOM_STATE, n_jobs=-1)
df_importancia = (pd.DataFrame({"variable": FEATURES, "importancia": importancia.importances_mean,
                                "desviacion": importancia.importances_std})
                  .sort_values("importancia", ascending=False).reset_index(drop=True))
df_importancia.to_csv(DIR_MODELOS / "importancias.csv", index=False)   # La app muestra este gráfico

fig, eje = plt.subplots(figsize=(9, 6))
eje.barh(df_importancia["variable"], df_importancia["importancia"], xerr=df_importancia["desviacion"], color="#4C72B0")
eje.invert_yaxis(); eje.set_xlabel("Disminución media del F1"); eje.set_title("Importancia por permutación (test)")
plt.tight_layout(); guardar_figura("importancia_permutacion"); plt.show()
df_importancia.round(4)   # Valores ≈ 0 (o negativos) = la variable no aporta al modelo

# %% [markdown]
# 🔎 **Interpretación:** Las variables más importantes por permutación (caída del F1 al desordenarlas) son:
#
# | Variable | Caída del F1 |
# |---|---|
# | Edad | 0,030 |
# | Jornada | 0,028 |
# | Educación del padre | 0,018 |
# | Educación de la madre | 0,017 |
# | Género | 0,010 |
# | Computador en el hogar | 0,008 |
# | Naturaleza del colegio | 0,008 |
# | Valle de Aburrá | 0,007 |
#
# El **estrato** aparece con una importancia baja (0,006), aunque en 3.6 estaba muy asociado al objetivo. Es la limitación descrita en el 📘: comparte información con la educación de los padres y las tenencias del hogar. Al desordenarlo, el modelo compensa con esas variables correlacionadas.
# Automóvil, zona rural/urbana, lavadora y colegio bilingüe tienen importancias de cero o negativas: no aportan al modelo una vez consideradas las demás.

# %% [markdown]
# #### 📘 Concepto: Equidad por subgrupos
# Un buen desempeño global puede esconder un desempeño pobre en algunos grupos. La **auditoría por subgrupos** calcula
# las métricas por separado para cada grupo (género, zona, naturaleza del colegio, subregión) y compara:
# - **Tasa real vs. tasa predicha:** si el modelo exagera o atenúa las diferencias que ya existen.
# - **Recall por grupo** (*igualdad de oportunidad*, Hardt et al., 2016): ¿detecta igual de bien a los estudiantes de alto
#   desempeño de cada grupo?
# - **Precision por grupo:** ¿se equivoca más al señalar "alto" en algún grupo?
#
# No existe una única definición de equidad y varias son incompatibles entre sí (Barocas, Hardt y Narayanan, 2023). El
# objetivo aquí es **hacer visibles** las diferencias para interpretarlas y discutirlas.

# %%
# 5.2.7 Desempeño por subgrupo: control de sesgo del modelo final (género, zona, naturaleza, subregión)
serie_pred = pd.Series(predicciones[NOMBRE_FINAL], index=X_test.index)   # Predicciones alineadas con los índices de test
filas = []
for var in ["estu_genero", "cole_area_ubicacion", "cole_naturaleza", "cole_valle_aburra"]:
    for grupo, indices in X_test.groupby(X_test[var].fillna("DESCONOCIDO")).groups.items():   # Índices de cada grupo
        real, pred = y_test.loc[indices], serie_pred.loc[indices]
        # zero_division=0 evita errores en grupos sin positivos predichos
        filas.append({"variable": var, "grupo": grupo, "registros": len(indices),
                      "tasa_real": real.mean(), "tasa_predicha": pred.mean(),
                      "precision": precision_score(real, pred, zero_division=0),
                      "recall": recall_score(real, pred, zero_division=0), "f1": f1_score(real, pred, zero_division=0)})
tabla_subgrupos = guardar_tabla(pd.DataFrame(filas).round(4), "desempeno_subgrupos")
tabla_subgrupos

# %% [markdown]
# 🔎 **Interpretación:** La auditoría por subgrupos revela **diferencias importantes**:
# - **Naturaleza del colegio:** F1 de 0,65 en colegios no oficiales frente a 0,34 en oficiales. En los oficiales el modelo solo detecta al 40 % de los estudiantes de alto desempeño (Recall 0,40, frente a 0,87 en los no oficiales).
# - **Subregión:** fuera del Valle de Aburrá, F1 0,38 frente a 0,50 dentro.
# - **Género:** el F1 es similar (0,47 en ambos), pero en hombres el modelo **sobrepredice** "alto" (35 % predicho frente a 20 % real), mientras en mujeres queda cerca de la tasa real (17 % frente a 15 %).
#
# El modelo funciona peor precisamente en los grupos menos favorecidos: los estudiantes de alto desempeño en colegios oficiales o fuera del área metropolitana tienen perfiles socioeconómicos "atípicos" que el modelo no reconoce. Es la amplificación del **sesgo histórico** anticipada en 2.4 y la razón principal para no usar este modelo en decisiones individuales.

# %%
# 5.2.8 Justificación técnica y estadística del modelo seleccionado (tabla 5.2 del informe)
fila_final = df_test.set_index("modelo").loc[NOMBRE_FINAL]   # Métricas de test del modelo final
ic_final = df_ic.set_index("candidato").loc[NOMBRE_FINAL]    # Su intervalo de confianza bootstrap
# El texto se arma con las cifras calculadas: si los datos cambian, la justificación cambia con ellos
justificacion_final = (
    f"Mayor F1 en validación cruzada ({cv_f1[NOMBRE_FINAL]:.4f}). En test: F1 = {fila_final['f1']:.4f} "
    f"(IC95 % bootstrap {ic_final['ic95_inf']:.3f}–{ic_final['ic95_sup']:.3f}), Recall = {fila_final['recall']:.4f}, "
    f"Precision = {fila_final['precision']:.4f}, ROC-AUC = {fila_final['roc_auc']:.4f}. "
    f"McNemar frente a {RIVAL}: p = {p_mcnemar:.3g}. t-test corregido (Nadeau–Bengio) en CV frente a {FAMILIA_RIVAL}: "
    f"p = {p_t_rival:.3g}; frente a la regresión logística base: p = {p_t_base:.3g}.")
guardar_tabla(pd.DataFrame({"modelo_final": [NOMBRE_FINAL], "justificacion": [justificacion_final]}), "modelo_seleccionado")
print(justificacion_final)

# %% [markdown]
# 🔎 **Interpretación:** La justificación final (tabla 5.2 del informe) reúne los argumentos técnicos y estadísticos:
# - El modelo tiene el mayor F1 en validación cruzada (0,4856).
# - En test obtiene F1 = 0,4730, con IC95 % de 0,453 a 0,494, Recall 0,58, Precision 0,40 y ROC-AUC 0,80.
# - McNemar indica que comete menos errores que Random Forest (p = 0,025).
# - El t-test corregido confirma que la diferencia de F1 frente a Random Forest no es significativa (p = 0,895).
#
# A igualdad de F1 prima la **simplicidad y la interpretabilidad**: la regresión logística se explica con coeficientes y pesa solo 20 KB.

# %% [markdown]
# ### 🧠 Para profundizar — Fase 5: Evaluación
# **Ideas clave**
# - Con clases desbalanceadas importan Precision, Recall, F1 y PR-AUC; la *accuracy* es secundaria.
# - Se **selecciona** con validación cruzada y se **confirma** con el test; usar el test para elegir sesga la estimación.
# - Una diferencia de métricas solo es relevante si sobrevive a una prueba estadística (bootstrap, McNemar, t corregido).
# - Interpretar el modelo (importancias) y auditarlo por grupos es parte de la evaluación, no un extra.
#
# **Preguntas de repaso**
# 1. Un modelo tiene Recall 0,80 y Precision 0,30. ¿Cuál es su F1 y qué significa para una secretaría que asigna becas?
# 2. ¿Por qué el t-test clásico sobre los folds de CV da p-valores demasiado pequeños?
# 3. ¿Qué indica una curva de aprendizaje donde entrenamiento y validación convergen en F1 ≈ 0,5?
# 4. Si el Recall es mucho menor en colegios rurales, ¿qué acciones de preparación o de modelamiento lo mitigarían?
#
# **Ejercicios**
# - Grafica Precision, Recall y F1 en test en función del umbral (0,05 a 0,95) y ubica el umbral elegido en 4.3.5.
# - Calcula la importancia por permutación con `scoring="roc_auc"` y compara el ranking con el de F1.
#
# **Referencias**
# - Saito, T. y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLOS ONE*, 10(3).
# - Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. *Neural Computation*, 10(7), 1895–1923.
# - Nadeau, C. y Bengio, Y. (2003). Inference for the generalization error. *Machine Learning*, 52, 239–281.
# - Efron, B. y Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
# - Molnar, C. (2022). *Interpretable Machine Learning* (2.ª ed.). https://christophm.github.io/interpretable-ml-book/
# - Hardt, M., Price, E. y Srebro, N. (2016). Equality of opportunity in supervised learning. *NeurIPS*.
# - Barocas, S., Hardt, M. y Narayanan, A. (2023). *Fairness and Machine Learning*. MIT Press.
