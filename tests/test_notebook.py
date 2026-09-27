"""Pruebas de la convención del notebook: comentarios en código, Markdown bajo cada celda y ejecución sin errores."""
from pathlib import Path

import nbformat
import pytest

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks" / "01_Material_de_estudio_Saber11.ipynb"
MARCADOR = "INTERPRETACION_PENDIENTE"


@pytest.fixture(scope="module")
def nb():
    assert NOTEBOOK.exists(), "Construya el notebook: .venv/Scripts/python scripts/construir_notebook.py"
    return nbformat.read(NOTEBOOK, as_version=4)


def celdas_codigo(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def test_cada_celda_de_codigo_tiene_markdown_debajo(nb):
    for i, celda in enumerate(nb.cells):
        if celda.cell_type == "code":
            assert i + 1 < len(nb.cells) and nb.cells[i + 1].cell_type == "markdown", f"Celda {i} sin Markdown debajo"


def es_comentario(linea):
    """Comentario de línea, comentario al final de una instrucción o línea de docstring."""
    return linea.startswith("#") or " # " in linea or '"""' in linea


def es_linea_de_datos(linea):
    """Literales puros (textos de diccionarios/listas) o cierres de paréntesis: no exigen comentario propio."""
    return linea[0] in "\"'" or linea.startswith(('f"', '("', '["')) or set(linea) <= set(")]}, ")


def test_codigo_comentado_didacticamente(nb):
    """Material de estudio: ≥ 2 comentarios por celda y ≥ 20 % de las líneas de código comentadas."""
    for celda in celdas_codigo(nb):
        lineas = [l.strip() for l in celda.source.splitlines() if l.strip()]
        comentarios = [l for l in lineas if es_comentario(l)]
        efectivas = [l for l in lineas if es_comentario(l) or not es_linea_de_datos(l)]
        densidad = len(comentarios) / max(1, len(efectivas))
        assert len(comentarios) >= 2 and densidad >= 0.20, f"Comentarios insuficientes ({densidad:.0%}):\n{celda.source[:80]}"


CONCEPTOS_OBLIGATORIOS = [
    "CRISP-DM", "Datos abiertos y API Socrata", "Diccionario de datos y fuga de información", "Calidad de datos",
    "Variable objetivo y desbalance", "Sesgos en los datos", "Selección de variables", "Perfilamiento automático",
    "Limpieza determinística", "Valores faltantes", "Partición entrenamiento/prueba", "Codificación y escalado",
    "Correlación de Pearson y Spearman", "V de Cramér e información mutua", "Análisis de Componentes Principales",
    "Balanceo de clases", "Validación cruzada y pipelines", "Modelos clásicos", "Métodos de ensamble",
    "Búsqueda de hiperparámetros", "Umbral de decisión", "Métricas de clasificación",
    "Comparación estadística de modelos", "Curva de aprendizaje", "Importancia por permutación",
    "Equidad por subgrupos", "Serialización de modelos", "Generalización a datos nuevos",
]


def test_celdas_de_concepto_presentes(nb):
    """Cada técnica clave tiene su celda 📘 de teoría antes del código."""
    textos = [c.source for c in nb.cells if c.cell_type == "markdown" and "📘 Concepto" in c.source]
    faltantes = [k for k in CONCEPTOS_OBLIGATORIOS if not any(f"📘 Concepto: {k}" in t for t in textos)]
    assert not faltantes, f"Faltan celdas de concepto: {faltantes}"


def test_cierre_de_cada_fase(nb):
    """Cada fase CRISP-DM termina con ideas clave, preguntas de repaso, ejercicios y referencias."""
    for fase in range(1, 7):
        cierre = [c.source for c in nb.cells if f"🧠 Para profundizar — Fase {fase}" in c.source]
        assert cierre, f"Falta el cierre de la fase {fase}"
        for seccion in ["Ideas clave", "Preguntas de repaso", "Ejercicios", "Referencias"]:
            assert seccion in cierre[0], f"Fase {fase}: falta '{seccion}'"


def test_notebook_ejecutado_sin_errores(nb):
    for celda in celdas_codigo(nb):
        assert celda.execution_count is not None, f"Celda sin ejecutar:\n{celda.source[:80]}"
        assert not any(o.get("output_type") == "error" for o in celda.outputs), f"Error en:\n{celda.source[:80]}"


def test_sin_interpretaciones_pendientes(nb):
    pendientes = [c.source[:60] for c in nb.cells if MARCADOR in c.source]
    assert not pendientes, f"{len(pendientes)} celdas con interpretación pendiente"
