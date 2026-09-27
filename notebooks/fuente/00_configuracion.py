# %% [markdown]
# # Proyecto Integrador — Metodología CRISP-DM
# ## Predicción de alto desempeño en la prueba Saber 11 · Antioquia 2022-2
#
# | Campo | Valor |
# |---|---|
# | Programa | Maestría en Ciencia de Datos — proyecto práctico de la asignatura |
# | Propósito | **Netamente educativo**: estudiar y profundizar en CRISP-DM y el aprendizaje supervisado |
# | Tipo de problema | Clasificación binaria (aprendizaje supervisado) |
# | Fuente | ICFES — *Resultados únicos Saber 11*, Datos Abiertos Colombia (`kgxf-xxbe`) |
# | Enlace | https://www.datos.gov.co/Educaci-n/Resultados-nicos-Saber-11/kgxf-xxbe |
#
# > ⚠️ **Aviso académico:** este cuaderno es material de estudio de nivel maestría. No es una herramienta oficial del ICFES
# > ni de ninguna entidad, y sus predicciones no deben usarse para tomar decisiones sobre estudiantes reales.
#
# ### Cómo estudiar este cuaderno
# | Símbolo | Qué es | Cómo usarlo |
# |---|---|---|
# | 📘 **Concepto** | Teoría de la técnica: definición, intuición, fórmula, supuestos y limitaciones | Léelo **antes** de ejecutar la celda |
# | 💻 Código comentado | Cada paso explica el *qué* y el *porqué* | Ejecútalo, lee los comentarios y modifica parámetros |
# | 🔎 **Interpretación** | Lectura de la salida con las cifras obtenidas | Compárala con tu propia lectura del resultado |
# | 🧠 **Para profundizar** | Ideas clave, preguntas de repaso, ejercicios y referencias al cierre de cada fase | Responde las preguntas sin mirar y haz los ejercicios |
#
# **Contenido:** 0. Configuración · 1. Negocio · 2. Datos · 3. Preparación · 4. Modelamiento · 5. Evaluación · 6. Despliegue

# %% [markdown]
# #### 📘 Concepto: CRISP-DM
# **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*, 1999–2000) es el proceso de referencia más usado para
# proyectos de analítica. Tiene **seis fases**:
#
# 1. **Entendimiento del negocio:** qué problema se resuelve, para quién y cómo se medirá el éxito.
# 2. **Entendimiento de los datos:** qué datos hay, qué significan y qué calidad tienen.
# 3. **Preparación de los datos:** selección, limpieza, transformación y construcción de variables (suele consumir 60–80 % del esfuerzo).
# 4. **Modelamiento:** elección y ajuste de algoritmos.
# 5. **Evaluación:** ¿el modelo cumple los objetivos de negocio, no solo los técnicos?
# 6. **Despliegue:** poner el modelo al servicio de los usuarios.
#
# El proceso es **iterativo**: lo que se descubre en una fase obliga a volver a otra (por ejemplo, un sesgo detectado en la
# evaluación lleva a revisar la preparación). Este cuaderno sigue las seis fases en orden y numera las secciones igual que el
# formato de entrega del curso.

# %% [markdown]
# ## 0. Configuración del entorno

# %%
# 0.1 Instalación de dependencias (solo se ejecuta en Google Colab)
# En local las librerías ya están en el entorno virtual .venv; en Colab hay que instalarlas en cada sesión.
import sys          # Permite saber qué intérprete de Python está corriendo y qué módulos están cargados
import subprocess   # Permite ejecutar "pip install" desde Python

# Colab carga el módulo google.colab al iniciar: si está presente, estamos en Colab
EN_COLAB = "google.colab" in sys.modules

# Versiones exactas usadas al entrenar (se inyectan desde requirements-notebook.txt al construir el cuaderno).
# Fijar versiones garantiza que otra persona obtenga los mismos resultados y que el modelo serializado cargue sin errores.
PAQUETES = {{PAQUETES}}
# Librerías que Colab puede tener ya cargadas en memoria (nombre del módulo -> nombre del paquete en pip)
LIBRERIAS_BASE = {"numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "matplotlib": "matplotlib",
                  "sklearn": "scikit-learn"}

if EN_COLAB:
    from importlib.metadata import version   # Lee la versión instalada de un paquete sin importarlo
    # Versiones de las librerías ya cargadas, antes de instalar
    antes = {paquete: version(paquete) for modulo, paquete in LIBRERIAS_BASE.items() if modulo in sys.modules}
    # "-q" (quiet) reduce la salida de pip; *PAQUETES pasa cada elemento de la lista como argumento independiente
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *PAQUETES])
    print("Dependencias instaladas en Colab:", PAQUETES)
    # Si pip cambió una librería que ya estaba en memoria, la sesión seguiría usando la versión vieja: hay que reiniciar
    cambiadas = [paquete for paquete, v in antes.items() if version(paquete) != v]
    if cambiadas:
        print(f"⚠️ pip cambió librerías que ya estaban cargadas: {', '.join(cambiadas)}. Reinicie la sesión (Entorno de "
              "ejecución → Reiniciar sesión) y vuelva a ejecutar desde esta celda: la segunda vez pip no cambiará nada.")
else:
    print("Ejecución local: se usa el entorno virtual .venv del proyecto.")

# %% [markdown]
# 🔎 **Interpretación:** En local el cuaderno usa el entorno virtual `.venv`, así que no instala nada. En Colab instalaría las versiones exactas de scikit-learn, imbalanced-learn, xgboost y ydata-profiling, más `setuptools<81`, que ydata-profiling necesita para importar `pkg_resources`. Si pip reemplaza una librería que Colab ya tenía cargada (por ejemplo numpy), la celda avisa que hay que reiniciar la sesión: seguir sin reiniciar mezclaría versiones en memoria.
# **Por qué importa:** el pipeline se serializa con una versión concreta de scikit-learn (fase 6), y cargarlo con otra versión puede fallar o cambiar las predicciones. Fijar versiones es la primera condición de la reproducibilidad.

# %%
# 0.2 Importación de librerías, agrupadas por propósito
# --- Utilidades estándar de Python ---
import json                           # Guardar la metadata del modelo en formato legible
import time                           # Medir cuánto tarda cada paso costoso
import warnings                       # Silenciar avisos que no afectan los resultados
from pathlib import Path              # Rutas de archivos independientes del sistema operativo
from urllib.parse import urlencode    # Construir la URL de la API con caracteres especiales codificados

# --- Manipulación de datos y visualización ---
import numpy as np                    # Cálculo numérico vectorizado
import pandas as pd                   # Tablas (DataFrame) y operaciones sobre columnas
import matplotlib.pyplot as plt       # Gráficos base
import seaborn as sns                 # Gráficos estadísticos sobre matplotlib
from scipy import stats               # Pruebas estadísticas (χ², t, binomial, Spearman)
from scipy.stats import loguniform, randint, uniform   # Distribuciones para la búsqueda aleatoria de hiperparámetros

# --- Librerías de modelamiento (se importa el módulo completo para registrar sus versiones) ---
import sklearn
import imblearn
import xgboost
import joblib                         # Serialización eficiente de modelos de scikit-learn
from sklearn.base import clone        # Copia "sin entrenar" de un estimador (evita reutilizar modelos ya ajustados)
from sklearn.compose import ColumnTransformer          # Aplica transformaciones distintas por tipo de columna
from sklearn.decomposition import PCA                  # Reducción de dimensiones
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier   # Ensambles
try:  # En algunas versiones la búsqueda por mitades sucesivas sigue siendo experimental y requiere habilitarse
    from sklearn.experimental import enable_halving_search_cv  # noqa: F401
except ImportError:
    pass
from sklearn.feature_selection import mutual_info_classif     # Información mutua (dependencia no lineal)
from sklearn.impute import SimpleImputer                       # Imputación de valores faltantes
from sklearn.inspection import permutation_importance         # Importancia de variables agnóstica al modelo
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, average_precision_score, classification_report,
                             f1_score, precision_score, recall_score, roc_auc_score)   # Métricas de clasificación
from sklearn.model_selection import (HalvingRandomSearchCV, RandomizedSearchCV, StratifiedKFold,
                                     TunedThresholdClassifierCV, cross_val_score, cross_validate,
                                     learning_curve, train_test_split)                 # Validación y búsqueda
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline                          # Encadena pasos de preprocesamiento
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler   # Codificación y escalado
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE                       # Sobremuestreo sintético de la clase minoritaria
from imblearn.pipeline import Pipeline as ImbPipeline          # Pipeline que admite pasos de remuestreo
from imblearn.under_sampling import RandomUnderSampler         # Submuestreo aleatorio de la clase mayoritaria
from xgboost import XGBClassifier                              # Boosting por gradiente optimizado

# Silenciamos avisos de deprecación y de categorías desconocidas: no cambian los resultados y ensucian la salida
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
pd.set_option("display.max_columns", 60)                 # Mostrar todas las columnas de las tablas anchas
pd.set_option("display.float_format", "{:.4f}".format)   # Cuatro decimales en las tablas

# Registrar versiones es parte de la reproducibilidad: un resultado sin versiones no se puede repetir con certeza
print(f"Python {sys.version.split()[0]} | pandas {pd.__version__} | numpy {np.__version__} | "
      f"scikit-learn {sklearn.__version__} | imbalanced-learn {imblearn.__version__} | xgboost {xgboost.__version__}")

# %% [markdown]
# 🔎 **Interpretación:** Todas las librerías cargan sin errores. Las versiones quedan registradas: Python 3.12.10, pandas 2.3.3, numpy 2.3.5, scikit-learn 1.9.1, imbalanced-learn 0.14.2 y xgboost 3.4.1.
# Se usa pandas 2.x, y no 3.x, porque ydata-profiling 4.18 exige `pandas<3`. Es un ejemplo real de **gestión de dependencias**: una librería del proyecto restringe la versión de otra.

# %%
# 0.3 Parámetros globales, rutas y utilidades
# Centralizar los parámetros aquí permite experimentar (p. ej. cambiar el umbral) sin buscar números "mágicos" en el código.
RANDOM_STATE = 42                                  # Semilla única: muestreos, particiones y modelos reproducibles
UMBRAL_ALTO_DESEMPENO = 300                        # Regla de negocio: punt_global >= 300 => clase positiva (1)
N_MUESTRA_MODELADO = 30_000                        # Muestra estratificada para modelar (el SVM escala ~n² en tiempo)
TEST_SIZE = 0.30                                   # 70 % entrenamiento / 30 % prueba, como exige la actividad
N_ITER_BUSQUEDA = 20                               # Combinaciones que prueba cada búsqueda aleatoria
FECHA_REFERENCIA = pd.Timestamp("2022-08-01")      # Referencia aproximada de la aplicación 2022-2 para calcular la edad
# StratifiedKFold conserva en cada fold la proporción de clases; shuffle + semilla => los mismos 5 folds para todos los modelos
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
np.random.seed(RANDOM_STATE)                       # Semilla también para funciones de numpy sin random_state propio

# Rutas relativas: funcionan igual en local (se ejecuta desde notebooks/) y en Colab (se ejecuta desde /content)
BASE_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
DIR_RAW = BASE_DIR / "data" / "raw"             # Datos tal como llegan de la fuente
DIR_PROC = BASE_DIR / "data" / "processed"      # Datos limpios
DIR_NUEVOS = BASE_DIR / "data" / "nuevos"       # Datos para probar el despliegue
DIR_MODELOS = BASE_DIR / "models"               # Pipeline serializado y metadata
DIR_REPORTES = BASE_DIR / "reports"             # Reporte HTML de perfilamiento
DIR_FIGURAS = DIR_REPORTES / "figures"          # Gráficos para el informe y la app
DIR_TABLAS = DIR_REPORTES / "tablas"            # Tablas que alimentan el informe U4
for carpeta in [DIR_RAW, DIR_PROC, DIR_NUEVOS, DIR_MODELOS, DIR_FIGURAS, DIR_TABLAS]:
    carpeta.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe (y sus padres)

sns.set_theme(style="whitegrid", palette="deep")   # Estilo visual uniforme para todos los gráficos
plt.rcParams["figure.dpi"] = 110                   # Resolución de las figuras en pantalla


def guardar_figura(nombre: str) -> None:
    """Guarda la figura activa en reports/figures para reutilizarla en el informe y la app."""
    plt.savefig(DIR_FIGURAS / f"{nombre}.png", bbox_inches="tight", dpi=150)


def guardar_tabla(df: pd.DataFrame, nombre: str) -> pd.DataFrame:
    """Exporta una tabla a reports/tablas (insumo del informe U4) y la devuelve para mostrarla."""
    df.to_csv(DIR_TABLAS / f"{nombre}.csv", index=False, encoding="utf-8")
    return df


print("Directorio base del proyecto:", BASE_DIR)

# %% [markdown]
# 🔎 **Interpretación:** Se fijan la semilla (42), el umbral de negocio (300), el tamaño de la muestra (30.000), la partición 70/30 y un único objeto `CV` con 5 folds estratificados.
# Que todos los modelos compartan exactamente los mismos folds es lo que permite, en la fase 5, comparar sus F1 fold a fold con pruebas **pareadas**. Las carpetas de salida se crean de forma relativa, así que el cuaderno funciona igual en local y en Colab.
