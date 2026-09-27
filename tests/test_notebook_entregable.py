"""Pruebas del notebook entregable (02): explicación antes y después de cada celda, ejecución, secciones y coherencia."""
import hashlib
import json
import re
from pathlib import Path

import nbformat
import pytest

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks" / "02_Entregable_Proyecto_Integrador_Saber11.ipynb"
METADATA = RAIZ / "models" / "metadata.json"
MARCADOR = "INTERPRETACION_PENDIENTE"
HUELLAS_EXCLUIDAS = {   # SHA-256 de las palabras que no deben aparecer (datos personales del autor del código)
    "5e4dd528bd5afe91002968564a82bca663f502adb51473c467c59d5229cf14fb",
    "dcd79ebce907b98a97ac818318c9e467731e44068412590ea34280dcadcf02d9",
}
SECCIONES_OBLIGATORIAS = [
    "1.1 Contexto", "1.2 Pregunta de negocio", "1.3 Regla de negocio", "1.4 Definición de las funciones", "1.5 Definición del criterio de negocio",
    "1.6 Objetivos", "2.1 Diccionario", "2.2 Reglas de calidad", "2.3 Distribución de la variable objetivo",
    "2.4 Análisis exploratorio", "2.5 Sesgos", "3.1 Selección de variables", "3.2 Descripción estadística",
    "3.3 Limpieza de errores", "3.4 Limpieza de nulos", "3.5 Codificación", "3.6 Relaciones", "3.7 Balanceo",
    "4.1 Modelos clásicos", "4.2 Modelos de ensamble", "4.3 Ajuste de hiperparámetros", "5.1 Resultados en test",
    "5.2 Modelo seleccionado", "5.3 Diagnóstico", "5.4 Equidad", "5.5 Criterio de negocio", "6.0 Serialización",
    "6.1 Predicción con datos nuevos", "6.2 Modelo prescriptivo", "6.3 Nivel de MLOps", "6.4 Aplicación",
    "7. Conclusiones", "Recomendaciones",
]


@pytest.fixture(scope="module")
def nb():
    assert NOTEBOOK.exists(), "Construya el notebook: .venv/Scripts/python scripts/construir_notebook.py --notebook entregable"
    return nbformat.read(NOTEBOOK, as_version=4)


def celdas_codigo(nb):
    return [c for c in nb.cells if c.cell_type == "code"]


def es_comentario(linea):
    return linea.startswith("#") or " # " in linea or '"""' in linea


def es_linea_de_datos(linea):
    return linea[0] in "\"'" or linea.startswith(('f"', '("', '["')) or set(linea) <= set(")]}, ")


def test_explicacion_antes_e_interpretacion_despues(nb):
    for i, celda in enumerate(nb.cells):
        if celda.cell_type == "code":
            assert i > 0 and "¿Qué hacemos y por qué?" in nb.cells[i - 1].source, f"Sin explicación previa: {celda.source[:60]}"
            assert i + 1 < len(nb.cells) and "🔎 **Interpretación:**" in nb.cells[i + 1].source, f"Sin interpretación: {celda.source[:60]}"


def test_codigo_comentado(nb):
    for celda in celdas_codigo(nb):
        lineas = [l.strip() for l in celda.source.splitlines() if l.strip()]
        comentarios = [l for l in lineas if es_comentario(l)]
        efectivas = [l for l in lineas if es_comentario(l) or not es_linea_de_datos(l)]
        densidad = len(comentarios) / max(1, len(efectivas))
        assert len(comentarios) >= 2 and densidad >= 0.20, f"Comentarios insuficientes ({densidad:.0%}):\n{celda.source[:80]}"


def test_ejecutado_sin_errores(nb):
    for celda in celdas_codigo(nb):
        assert celda.execution_count is not None, f"Celda sin ejecutar:\n{celda.source[:80]}"
        assert not any(o.get("output_type") == "error" for o in celda.outputs), f"Error en:\n{celda.source[:80]}"


def test_sin_interpretaciones_pendientes(nb):
    assert not [c for c in nb.cells if MARCADOR in c.source], "Hay interpretaciones pendientes"


def test_secciones_obligatorias(nb):
    titulos = [t for c in nb.cells if c.cell_type == "markdown" for t in re.findall(r"^#+\s+(.*)$", c.source, re.M)]
    faltantes = [s for s in SECCIONES_OBLIGATORIAS if not any(t.startswith(s) for t in titulos)]
    assert not faltantes, f"Faltan secciones: {faltantes}"


def test_graficos_suficientes(nb):
    figuras = sum(1 for c in celdas_codigo(nb) for o in c.outputs if "image/png" in o.get("data", {}))
    assert figuras >= 30, f"Solo {figuras} gráficos"


def test_modelo_final_coherente_con_el_desplegado(nb):
    textos = [o.get("text", "") for c in celdas_codigo(nb) for o in c.outputs]
    linea = next(t for t in textos if "Modelo final:" in t)
    nombre = re.search(r"Modelo final: (.+?) \|", linea).group(1)
    assert nombre == json.loads(METADATA.read_text(encoding="utf-8"))["modelo_final"]


def test_sin_datos_personales_del_autor_del_codigo(nb):
    # Se comparan huellas SHA-256 de cada palabra para no escribir en el repositorio los datos que se quieren excluir
    textos = [c.source for c in nb.cells] + [o.get("text", "") for c in celdas_codigo(nb) for o in c.outputs]
    palabras = set(re.findall(r"[a-záéíóúüñ]+", "\n".join(textos).lower()))
    huellas = {hashlib.sha256(p.encode("utf-8")).hexdigest() for p in palabras}
    assert not huellas & HUELLAS_EXCLUIDAS, "El notebook contiene datos personales del autor del código"
