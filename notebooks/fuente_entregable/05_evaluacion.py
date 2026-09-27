# %% [markdown]
# ## 5. Evaluación
# ### 5.1 Resultados en test
#
# **Métricas utilizadas y su justificación.** El objetivo es identificar a los estudiantes de **alto desempeño** (clase
# minoritaria, ~17 %). Por eso las métricas se centran en la clase positiva:
#
# | Métrica | Qué mide | Fórmula | Implicación práctica en este caso |
# |---|---|---|---|
# | **Accuracy** | Aciertos sobre el total | (VP+VN)/N | Engañosa con desbalance: decir siempre "no alto" da ~83 % |
# | **Precision** | De los señalados como "alto", cuántos lo son | VP/(VP+FP) | Evita dirigir becas o estímulos de alto desempeño a estudiantes que no alcanzarán ese nivel |
# | **Recall** | De los que son "alto", cuántos se detectan | VP/(VP+FN) | Evita dejar por fuera a estudiantes con potencial |
# | **F1** (principal) | Equilibrio entre Precision y Recall | 2·P·R/(P+R) | Solo es alta si ambas lo son |
# | **ROC-AUC** | Capacidad de ordenar positivos por encima de negativos | Área bajo la curva ROC | No depende del umbral |
# | **PR-AUC** | Precision media a lo largo del Recall | Área bajo la curva PR | Más informativa con clases desbalanceadas (el azar vale ~0,17) |

# %% [markdown]
# **¿Qué hacemos y por qué?** Evaluamos **por primera y única vez** en el 30 % de prueba los 14 candidatos: 8 modelos
# base, 3 ajustados y esos mismos 3 con el umbral óptimo.

# %%
#@copiar 5.1.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra las métricas de todos los modelos en el conjunto de prueba (9.000 estudiantes nunca usados). Los tres modelos con umbral ajustado quedan arriba:
#
# | Modelo | F1 | Precision | Recall | Accuracy | ROC-AUC |
# |---|---|---|---|---|---|
# | Regresión Logística + umbral | 0,4730 | 0,400 | 0,580 | 0,779 | 0,798 |
# | Random Forest + umbral | 0,4726 | 0,392 | 0,596 | 0,772 | 0,797 |
# | SVM + umbral | 0,4707 | 0,385 | 0,607 | 0,766 | 0,780 |
#
# En general, sin ajustar el umbral el Recall es mayor (~0,74) pero la Precision cae a ~0,33: el umbral cambia el **punto de operación**, no la capacidad de ordenar (la ROC-AUC de la logística es 0,7975 con y sin umbral). Los resultados en prueba son muy parecidos a los de validación cruzada (0,473 frente a 0,486), así que la CV fue una buena estimación.

# %% [markdown]
# **¿Qué hacemos y por qué?** Comparamos visualmente el F1 en test de todos los candidatos; los ajustados van en rojo.

# %%
#@copiar 5.1.2

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra el F1 en prueba de todos los modelos, ordenado de mayor a menor.
# En general, se separan tres grupos: arriba, los modelos con umbral ajustado (F1 ≈ 0,47); en el medio, los modelos base y ajustados sin umbral de la familia fuerte, XGBoost y la votación (0,45–0,46); abajo, el bagging, el árbol y KNN (0,42–0,43). El salto más grande no viene de cambiar de algoritmo, sino de **ajustar el umbral**.

# %% [markdown]
# ### 5.2 Modelo seleccionado

# %% [markdown]
# **¿Qué hacemos y por qué?** Elegimos el modelo por su F1 en **validación cruzada**, no por el de test, para no sesgar la
# estimación. Identificamos además al rival más fuerte de otra familia de modelos.

# %%
#@copiar 5.2.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra los candidatos ordenados por su F1 en validación cruzada. El mejor es la **Regresión Logística (ajustado + umbral)**, con 0,4856; el rival más fuerte de otra familia es Random Forest (ajustado + umbral), con 0,4815.
# En general, el orden por validación cruzada coincide con el orden en prueba (0,4730 frente a 0,4726), así que la selección es **consistente**. Elegir con la CV y solo confirmar con la prueba evita escoger el modelo que "tuvo suerte" en un único conjunto.

# %% [markdown]
# **¿Qué hacemos y por qué?** Justificamos la elección **estadísticamente** con tres pruebas:
# - Intervalo de confianza bootstrap del F1.
# - Prueba de McNemar sobre los errores en test.
# - t-test corregido de Nadeau–Bengio sobre los folds de validación cruzada.

# %%
#@copiar 5.2.2

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra tres pruebas estadísticas complementarias:
# - **IC bootstrap al 95 %:** logística 0,453–0,494; Random Forest 0,452–0,492; SVM 0,451–0,491. Los intervalos se **solapan casi por completo**: en F1 los tres modelos son estadísticamente indistinguibles.
# - **McNemar (logística frente a Random Forest):** de los 669 casos en que discrepan, la logística acierta 364 y Random Forest 305 (p = 0,025). La logística comete **significativamente menos errores totales** (accuracy 0,779 frente a 0,772), aunque su F1 sea igual.
# - **t-test corregido de Nadeau–Bengio en CV:** frente a Random Forest, p = 0,895; frente a la logística base, p = 0,670. No hay diferencia real de F1, y ajustar `C` no cambió el F1 de forma significativa: la ganancia vino del umbral.
#
# En general, con F1 equivalente, menos errores totales, un archivo de 20 KB y coeficientes interpretables, la **regresión logística es la mejor elección** (principio de parsimonia).

# %% [markdown]
# **¿Qué hacemos y por qué?** Redactamos la justificación técnica y estadística del modelo final con las cifras obtenidas
# (tabla 5.2 del informe).

# %%
#@copiar 5.2.8 como 5.2.3

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra la justificación final del modelo seleccionado (tabla 5.2 del informe):
# - Tiene el mayor F1 en validación cruzada (0,4856).
# - En prueba obtiene F1 = 0,4730 (IC95 % de 0,453 a 0,494), Recall 0,58, Precision 0,40 y ROC-AUC 0,80.
# - McNemar indica que comete menos errores que Random Forest (p = 0,025).
# - El t-test corregido confirma que su diferencia de F1 con Random Forest no es significativa (p = 0,895).
#
# En general, a igualdad de F1 prima la **simplicidad y la interpretabilidad**: la regresión logística se explica con coeficientes (5.3.8) y pesa solo 20 KB.

# %% [markdown]
# ### 5.3 Diagnóstico gráfico del modelo final
# En regresión se analizan los valores reales frente a los predichos y los residuos. En clasificación, los equivalentes son la
# matriz de confusión, las curvas ROC y PR, la distribución de probabilidades por clase, el efecto del umbral, la ganancia
# acumulada y la calibración.

# %% [markdown]
# **¿Qué hacemos y por qué?** La **matriz de confusión** muestra cuántos estudiantes clasifica bien y mal el modelo en cada
# clase: verdaderos y falsos positivos, verdaderos y falsos negativos.

# %%
#@copiar 5.2.3 como 5.3.1

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la matriz de confusión del modelo final en prueba:
#
# | | Predicho alto | Predicho no alto |
# |---|---|---|
# | **Real alto** (1.541) | 893 detectados (VP) | 648 no detectados (FN) |
# | **Real no alto** (7.459) | 1.342 falsos positivos (FP) | 6.117 correctos (VN) |
#
# En general, de cada 10 estudiantes señalados como "alto" 4 lo son (Precision 0,40), y se detecta al 58 % de quienes realmente lo son (Recall 0,58). Para la clase "no alto" el desempeño es alto (F1 0,86), porque es la clase mayoritaria. Los falsos positivos son estudiantes con buen perfil socioeconómico que no alcanzaron 300 puntos, y los falsos negativos son estudiantes que superaron las expectativas de su contexto: ambos muestran los límites de predecir solo con variables estructurales.

# %% [markdown]
# **¿Qué hacemos y por qué?** Comparamos cada métrica del modelo con la que obtendría un clasificador **al azar** que marca
# "alto" con la misma frecuencia de la población. Es la línea base mínima que cualquier modelo útil debe superar.

# %%
# 5.3.2 Métricas del modelo final frente a un clasificador aleatorio que respeta la prevalencia
metricas_final = df_test.set_index("modelo").loc[NOMBRE_FINAL, ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
prevalencia = y_test.mean()                                                # Proporción real de alto desempeño en test
# Valores esperados de un clasificador que marca "alto" al azar con probabilidad igual a la prevalencia
referencia_azar = pd.Series({"accuracy": prevalencia**2 + (1 - prevalencia)**2, "precision": prevalencia,
                             "recall": prevalencia, "f1": prevalencia, "roc_auc": 0.5, "pr_auc": prevalencia})
comparacion_azar = pd.DataFrame({"Modelo final": metricas_final.astype(float), "Azar (prevalencia)": referencia_azar})
fig, eje = plt.subplots(figsize=(11, 4.5))
comparacion_azar.plot.bar(ax=eje, color=["#1d4ed8", "#9ca3af"], rot=0)
for contenedor in eje.containers:
    eje.bar_label(contenedor, fmt="%.3f", fontsize=8)                      # Valor encima de cada barra
eje.set_ylim(0, 1); eje.set_ylabel("Valor de la métrica"); eje.set_title("Modelo final frente al azar (conjunto de prueba)")
plt.tight_layout(); guardar_figura("e_metricas_vs_azar"); plt.show()
comparacion_azar.round(4)

# %% [markdown]
# 🔎 **Interpretación:** El gráfico y la tabla comparan las métricas del modelo final en prueba con las de un clasificador aleatorio que marca "alto" con la misma frecuencia que la prevalencia (17,1 %).
# En general, el modelo supera al azar en todas las métricas, pero la ganancia es muy distinta según cuál se mire:
# - **Accuracy:** 0,779 frente a 0,716. La diferencia es pequeña porque el azar ya acierta mucho al repetir la clase mayoritaria; por eso la accuracy no sirve para evaluar este problema.
# - **Precision y Recall:** 0,400 y 0,579 frente a 0,171. El modelo encuentra estudiantes de alto desempeño con una tasa 2,3 veces mayor que el azar.
# - **ROC-AUC y PR-AUC:** 0,798 frente a 0,500 y 0,461 frente a 0,171, respectivamente.
#
# Esta comparación da contexto a un F1 de 0,473, que aislado podría parecer bajo: frente a la línea de base del azar (0,171), el modelo casi la triplica.

# %% [markdown]
# **¿Qué hacemos y por qué?** Las **curvas ROC y Precisión–Recall** evalúan la capacidad del modelo de ordenar a los
# estudiantes en todos los umbrales posibles, no solo en el elegido.

# %%
#@copiar 5.2.4 como 5.3.3

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra las curvas ROC (izquierda) y Precisión-Recall (derecha) de los tres finalistas en prueba.
# En general:
# - **ROC:** la logística y Random Forest son casi idénticas (AUC ≈ 0,80) y el SVM queda algo por debajo (0,78).
# - **Precisión-Recall:** Random Forest tiene la mejor PR-AUC (0,48), seguida de la logística (0,46); el SVM queda en 0,40. Todas superan con amplitud la línea del azar (prevalencia 0,17).
#
# Random Forest es algo mejor en la zona de **alta precisión y bajo recall**, útil si solo interesara identificar a los estudiantes con probabilidad muy alta. En el punto de operación elegido (F1 máximo), ambos modelos coinciden.

# %% [markdown]
# **¿Qué hacemos y por qué?** Graficamos la distribución de la probabilidad predicha según la clase real, el equivalente
# del **gráfico de residuos**. Si el modelo separa bien, los "alto" reales se concentran a la derecha del umbral y los "no
# alto" a la izquierda.

# %%
# 5.3.4 Distribución de la probabilidad predicha según la clase real (equivalente a los residuos en regresión)
prob_test = puntaje_positivo(modelo_final, X_test)                         # Probabilidad de alto desempeño en test
UMBRAL_MODELO = float(getattr(modelo_final, "best_threshold_", 0.5))       # Umbral con el que decide el modelo final
fig, eje = plt.subplots(figsize=(11, 4.5))
for clase, color, nombre in [(0, "#9ca3af", "Real: no alto"), (1, "#dc2626", "Real: alto")]:
    sns.histplot(prob_test[y_test.to_numpy() == clase], bins=40, stat="density", element="step",
                 fill=True, alpha=0.3, color=color, label=nombre, ax=eje)  # Densidad: compara formas, no tamaños
eje.axvline(UMBRAL_MODELO, color="black", ls="--", label=f"Umbral = {UMBRAL_MODELO:.3f}")
eje.set_xlabel("Probabilidad predicha de alto desempeño"); eje.set_title("¿Separa el modelo las dos clases?"); eje.legend()
plt.tight_layout(); guardar_figura("e_distribucion_probabilidades"); plt.show()
# Resumen de la probabilidad predicha por clase real
pd.DataFrame({"probabilidad": prob_test, "clase_real": y_test.to_numpy()}).groupby("clase_real")["probabilidad"].describe().round(3)

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la distribución de la probabilidad predicha por el modelo para los estudiantes que realmente son "no alto" (gris) y "alto" (rojo), con el umbral de 0,615 marcado. Es el equivalente, en clasificación, al análisis de residuos de una regresión.
# En general:
# - Los estudiantes de alto desempeño reciben probabilidades claramente mayores: mediana 0,668 frente a 0,342.
# - Las dos distribuciones **se solapan mucho** entre 0,35 y 0,75: allí conviven perfiles socioeconómicos parecidos con resultados distintos. Ese solapamiento es el límite de lo que se puede predecir con estas variables.
# - A la derecha del umbral queda la mayoría de los "alto" (58 %, el Recall), pero también una cola de "no alto" que forma los falsos positivos.
#
# Mover el umbral solo cambia qué parte del solapamiento se acepta; no elimina el error.

# %% [markdown]
# **¿Qué hacemos y por qué?** Mostramos cómo cambian Precision, Recall y F1 al mover el umbral de decisión. Así se explica
# por qué el umbral elegido en validación cruzada no es 0,5.

# %%
# 5.3.5 Precision, Recall y F1 según el umbral de decisión
precision_c, recall_c, umbrales_c = precision_recall_curve(y_test, prob_test)
f1_c = 2 * precision_c * recall_c / np.clip(precision_c + recall_c, 1e-12, None)   # F1 en cada umbral (sin dividir por 0)
fig, eje = plt.subplots(figsize=(11, 4.5))
eje.plot(umbrales_c, precision_c[:-1], label="Precision")                  # La curva trae un punto más que umbrales
eje.plot(umbrales_c, recall_c[:-1], label="Recall")
eje.plot(umbrales_c, f1_c[:-1], label="F1", lw=2.5)
eje.axvline(UMBRAL_MODELO, color="black", ls="--", label=f"Umbral elegido en CV = {UMBRAL_MODELO:.3f}")
eje.axvline(0.5, color="gray", ls=":", label="Umbral por defecto = 0,5")
eje.set_xlabel("Umbral"); eje.set_ylabel("Valor"); eje.set_title("Compromiso Precision–Recall según el umbral (test)"); eje.legend()
plt.tight_layout(); guardar_figura("e_metricas_vs_umbral"); plt.show()
mejor_i = int(np.nanargmax(f1_c[:-1]))                                     # Umbral que maximizaría F1 en test (solo referencia)
print(f"F1 máximo posible en test = {f1_c[mejor_i]:.4f} (umbral {umbrales_c[mejor_i]:.3f}); "
      f"con el umbral elegido en CV = {f1_score(y_test, prob_test >= UMBRAL_MODELO):.4f}")

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra cómo cambian Precision, Recall y F1 en prueba al mover el umbral de decisión entre 0 y 1, con el umbral por defecto (0,5) y el elegido en validación cruzada (0,615).
# En general, al subir el umbral la Precision aumenta y el Recall disminuye, y el F1 forma una **meseta amplia** entre 0,5 y 0,7. El F1 máximo posible en prueba sería 0,4745, con un umbral de 0,619, casi igual al 0,4730 obtenido con el umbral elegido solo con entrenamiento (0,615).
# Que el umbral de la validación cruzada caiga tan cerca del óptimo de prueba confirma que se eligió sin sobreajuste. En la práctica, el umbral podría moverse dentro de la meseta según la prioridad del programa (más Recall para no dejar candidatos por fuera, o más Precision para concentrar recursos) casi sin perder F1.

# %% [markdown]
# **¿Qué hacemos y por qué?** La **curva de ganancia acumulada** responde a la pregunta del negocio: si se revisa primero
# al X % de estudiantes con mayor probabilidad, ¿qué % de los de alto desempeño se capta? El *lift* compara ese resultado
# con una selección al azar.

# %%
# 5.3.6 Curva de ganancia acumulada y tabla de lift
orden = np.argsort(-prob_test)                                             # Estudiantes de mayor a menor probabilidad
positivos_ordenados = y_test.to_numpy()[orden]
fraccion_poblacion = np.arange(1, len(orden) + 1) / len(orden)             # % de estudiantes revisados
fraccion_captada = np.cumsum(positivos_ordenados) / positivos_ordenados.sum()   # % de alto desempeño captado
fig, eje = plt.subplots(figsize=(8, 6))
eje.plot(fraccion_poblacion, fraccion_captada, lw=2.5, label="Modelo final")
eje.plot([0, 1], [0, 1], "k--", label="Selección al azar")
eje.plot([0, prevalencia, 1], [0, 1, 1], color="gray", ls=":", label="Modelo perfecto")   # Capta todo al revisar la prevalencia
eje.set_xlabel("Fracción de estudiantes revisados (ordenados por probabilidad)"); eje.set_ylabel("Fracción de alto desempeño captada")
eje.set_title("Curva de ganancia acumulada (test)"); eje.legend()
plt.tight_layout(); guardar_figura("e_ganancia_acumulada"); plt.show()
# Tabla de lift: cuántas veces mejor que el azar es revisar el top X %
filas = []
for p in [0.05, 0.10, 0.20, 0.30, 0.50]:
    captado = fraccion_captada[int(np.ceil(p * len(orden))) - 1]
    filas.append({"top_pct_revisado": int(p * 100), "pct_alto_captado": round(100 * captado, 1), "lift": round(captado / p, 2)})
guardar_tabla(pd.DataFrame(filas), "e_ganancia_lift")

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la curva de ganancia acumulada: qué fracción de los estudiantes de alto desempeño se capta al revisar a los estudiantes ordenados de mayor a menor probabilidad. La tabla resume el *lift* en varios cortes.
#
# | Estudiantes revisados | % de "alto" captado | Lift |
# |---|---|---|
# | 5 % | 18,8 % | 3,76 |
# | 10 % | 31,9 % | 3,19 |
# | 20 % | 50,7 % | 2,54 |
# | 30 % | 64,7 % | 2,16 |
# | 50 % | 84,3 % | 1,69 |
#
# En general, revisando solo al 20 % de los estudiantes con mayor probabilidad se encuentra a la mitad de los de alto desempeño: 2,5 veces más de lo que daría una selección al azar. El lift es mayor en los primeros cortes, así que el modelo es más útil cuando los recursos son escasos y solo se puede atender a una fracción pequeña.

# %% [markdown]
# **¿Qué hacemos y por qué?** La **curva de calibración** compara la probabilidad predicha con la frecuencia real de alto
# desempeño en 10 grupos. Muestra si "70 %" significa realmente 70 %.

# %%
# 5.3.7 Curva de calibración de las probabilidades del modelo final
frac_real, prob_media = calibration_curve(y_test, prob_test, n_bins=10, strategy="quantile")   # 10 grupos de igual tamaño
fig, eje = plt.subplots(figsize=(7, 6))
eje.plot(prob_media, frac_real, marker="o", lw=2, label="Modelo final")
eje.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")               # Diagonal: probabilidad = frecuencia real
eje.set_xlabel("Probabilidad predicha media (por grupo)"); eje.set_ylabel("Fracción real de alto desempeño")
eje.set_title("Curva de calibración (test)"); eje.legend()
plt.tight_layout(); guardar_figura("e_calibracion"); plt.show()
# Probabilidad predicha frente a la tasa observada en cada grupo
pd.DataFrame({"prob_predicha_media": prob_media, "tasa_real": frac_real}).round(3)

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la curva de calibración: para grupos de estudiantes con probabilidad predicha parecida, compara esa probabilidad media con la fracción que realmente tuvo alto desempeño.
# En general, la curva queda **por debajo de la diagonal** en todo el rango: el modelo **sobreestima** la probabilidad. Por ejemplo, el grupo con probabilidad media 0,61 tiene una tasa real del 24 %, y el de 0,85 una del 55 %.
# Es la consecuencia esperada del submuestreo: el modelo aprendió con clases 50/50 y "cree" que el alto desempeño es mucho más común que el 17 % real. El **orden** de los estudiantes sí es correcto (la curva siempre sube), y por eso el F1 y el ranking no se ven afectados. Si las probabilidades se fueran a comunicar como riesgos, habría que recalibrarlas (por ejemplo, con `CalibratedClassifierCV`); para clasificar con un umbral ajustado no es necesario.

# %% [markdown]
# **¿Qué hacemos y por qué?** Si el modelo final es lineal (regresión logística), sus **coeficientes** muestran la dirección
# y la magnitud del efecto de cada variable codificada. El *odds ratio* = e^coef indica cuánto se multiplican las odds de
# alto desempeño.

# %%
# 5.3.8 Coeficientes del modelo final: los 10 efectos más positivos y los 10 más negativos
modelo_interno = getattr(modelo_final, "estimator_", modelo_final)         # Pipeline dentro del ajuste de umbral
clasificador = modelo_interno.named_steps["modelo"]
if hasattr(clasificador, "coef_"):                                          # Solo los modelos lineales tienen coeficientes
    nombres = modelo_interno.named_steps["preprocesamiento"].get_feature_names_out()
    coeficientes = pd.DataFrame({"variable": nombres, "coeficiente": clasificador.coef_[0]})
    coeficientes["odds_ratio"] = np.exp(coeficientes["coeficiente"])       # Factor que multiplica las odds de alto desempeño
    extremos = pd.concat([coeficientes.nlargest(10, "coeficiente"), coeficientes.nsmallest(10, "coeficiente")])
    fig, eje = plt.subplots(figsize=(10, 8))
    colores = ["#16a34a" if c > 0 else "#dc2626" for c in extremos["coeficiente"]]   # Verde aumenta, rojo disminuye
    eje.barh(extremos["variable"], extremos["coeficiente"], color=colores)
    eje.axvline(0, color="black", lw=1); eje.invert_yaxis()
    eje.set_xlabel("Coeficiente (log-odds)"); eje.set_title("Efectos más positivos (verde) y más negativos (rojo)")
    plt.tight_layout(); guardar_figura("e_coeficientes"); plt.show()
    guardar_tabla(coeficientes.round(4), "e_coeficientes")
    display(extremos.round(3))
else:
    print("El modelo final no es lineal: sus efectos se leen con la importancia por permutación (5.3.9).")

# %% [markdown]
# 🔎 **Interpretación:** El gráfico y la tabla muestran los 10 coeficientes más positivos y los 10 más negativos de la regresión logística, con su *odds ratio* (e^coeficiente): cuánto se multiplican las odds de alto desempeño cuando la columna pasa de 0 a 1 (o aumenta una desviación estándar, en las variables estandarizadas).
# En general:
# - **Jornada:** es el factor más fuerte en ambos sentidos. La jornada completa multiplica las odds por 2,0, mientras la sabatina (0,43) y la nocturna (0,49) las reducen a menos de la mitad.
# - **Edad:** cada desviación estándar adicional (≈ 1,7 años) reduce las odds a 0,55: la extraedad es la señal individual más clara.
# - **Hogar:** la educación de la madre (1,35) y la del padre (1,32) por desviación estándar, tener internet (1,42) y no tener computador (0,67) confirman el gradiente socioeconómico.
# - **Género:** ser hombre aumenta las odds (1,32) y ser mujer las reduce (0,74), en línea con la brecha observada en 2.5.
#
# Algunos coeficientes deben leerse con cuidado. `cole_bilingue_N` positivo y `cole_bilingue_S` negativo parecen contradecir el sentido común, pero el grupo bilingüe es diminuto (0,3 %) y su efecto ya lo capturan el estrato y la naturaleza del colegio. En una codificación one-hot los coeficientes son **relativos** entre categorías de la misma variable, y con variables correlacionadas se reparten el efecto. Son asociaciones condicionadas a las demás variables, no efectos causales.

# %% [markdown]
# **¿Qué hacemos y por qué?** La **importancia por permutación** mide cuánto cae el F1 al desordenar cada variable original.
# Sirve para cualquier modelo y habla en la métrica del negocio.

# %%
#@copiar 5.2.6 como 5.3.9

# %% [markdown]
# 🔎 **Interpretación:** La tabla y el gráfico muestran la importancia por permutación (caída del F1 en prueba al desordenar cada variable):
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
# En general, el **estrato** aparece con una importancia baja (0,006) aunque en 3.6 estaba muy asociado al objetivo: comparte información con la educación de los padres y las tenencias del hogar, así que al desordenarlo el modelo compensa con esas variables correlacionadas. Automóvil, zona, lavadora y colegio bilingüe tienen importancias de cero o negativas: no aportan una vez consideradas las demás.

# %% [markdown]
# **¿Qué hacemos y por qué?** La **curva de aprendizaje** muestra si más datos mejorarían el modelo. Justifica haber
# modelado con 30.000 registros.

# %%
#@copiar 5.2.5 como 5.3.10

# %% [markdown]
# 🔎 **Interpretación:** El gráfico muestra la curva de aprendizaje del modelo final: F1 de entrenamiento y de validación según el número de registros usados.
# En general, ambas curvas **convergen alrededor de 0,47** a partir de unos 10 mil registros, y con 17 mil la brecha es prácticamente nula. Es un patrón de **alto sesgo**, no de alta varianza: agregar más registros (por ejemplo, los 72 mil disponibles) apenas mejoraría el modelo, lo que justifica haber modelado con 30 mil. Para mejorar habría que aportar **variables más informativas**, como el efecto colegio.

# %% [markdown]
# ### 5.4 Equidad por subgrupos

# %% [markdown]
# **¿Qué hacemos y por qué?** Calculamos las métricas del modelo final por género, zona, naturaleza y subregión. Un buen
# desempeño global puede esconder grupos donde el modelo funciona peor.

# %%
#@copiar 5.2.7 como 5.4.1

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra las métricas del modelo final por subgrupo en el conjunto de prueba: tasa real y predicha de alto desempeño, Precision, Recall y F1.
# En general, hay **diferencias importantes**:
# - **Naturaleza del colegio:** F1 de 0,65 en colegios no oficiales frente a 0,34 en oficiales. En los oficiales el modelo solo detecta al 40 % de los estudiantes de alto desempeño (Recall 0,40, frente a 0,87 en los no oficiales).
# - **Subregión:** fuera del Valle de Aburrá el F1 es 0,38, frente a 0,50 dentro.
# - **Zona:** el F1 es parecido (0,48 rural y 0,47 urbano), pero en la zona urbana el modelo predice "alto" para el 28 % frente a un 18 % real.
# - **Género:** el F1 es similar (0,47 en ambos), pero en hombres el modelo **sobrepredice** "alto" (35 % predicho frente a 20 % real), mientras en mujeres queda cerca de la tasa real (17 % frente a 15 %).
#
# El modelo funciona peor precisamente en los grupos menos favorecidos, cuyos estudiantes de alto desempeño tienen perfiles socioeconómicos "atípicos" que el modelo no reconoce. Es la amplificación del **sesgo histórico** descrito en 2.5 y la razón principal para no usar el modelo en decisiones individuales.

# %% [markdown]
# ### 5.5 Criterio de negocio: ¿se cumple?

# %% [markdown]
# **¿Qué hacemos y por qué?** Verificamos uno por uno los tres criterios de negocio definidos en 1.5 con los resultados del
# conjunto de prueba.

# %%
# 5.5.1 Verificación del criterio de negocio definido en 1.5
fila_final = df_test.set_index("modelo").loc[NOMBRE_FINAL]                  # Métricas de test del modelo final
f1_linea_base = df_test.set_index("modelo").loc["Regresión Logística (base)", "f1"]
lift = fila_final["precision"] / prevalencia                                # Veces que el modelo supera a la selección al azar
criterios = pd.DataFrame([
    {"criterio": "Detectar al menos la mitad de los estudiantes de alto desempeño", "meta": "Recall ≥ 0,55",
     "resultado": round(fila_final["recall"], 4), "cumple": fila_final["recall"] >= 0.55},
    {"criterio": "Focalizar mejor que el azar", "meta": f"Lift ≥ 2 (tasa base = {prevalencia:.3f})",
     "resultado": round(lift, 2), "cumple": lift >= 2},
    {"criterio": "Superar la línea base", "meta": f"F1 > {f1_linea_base:.4f} (regresión logística base)",
     "resultado": round(fila_final["f1"], 4), "cumple": fila_final["f1"] > f1_linea_base}])
criterios["cumple"] = criterios["cumple"].map({True: "✅ Sí", False: "❌ No"})    # Lectura directa para el informe
guardar_tabla(criterios, "e_criterio_negocio")
criterios

# %% [markdown]
# 🔎 **Interpretación:** La tabla muestra la verificación de los tres criterios de negocio definidos en 1.5 con el modelo final en el conjunto de prueba:
#
# | Criterio | Meta | Resultado | ¿Cumple? |
# |---|---|---|---|
# | Detectar al menos la mitad de los estudiantes de alto desempeño | Recall ≥ 0,55 | 0,580 | ✅ Sí |
# | Focalizar mejor que el azar | Lift ≥ 2 | 2,33 | ✅ Sí |
# | Superar la línea base | F1 > 0,4593 | 0,4730 | ✅ Sí |
#
# En general, **los tres criterios se cumplen**, aunque con márgenes modestos: el Recall supera la meta en 3 puntos y el lift en 0,33. El modelo es útil para **focalizar** programas (priorizar a quién ofrecer acompañamiento o preparación), pero no para decidir sobre estudiantes individuales, por los errores descritos en 5.3 y las diferencias por grupo de 5.4.
