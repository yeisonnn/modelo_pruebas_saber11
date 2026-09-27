"""Construye los notebooks del proyecto desde sus fuentes py:percent y opcionalmente los ejecuta.

Uso:
  python scripts/construir_notebook.py                                   # notebook de estudio (01)
  python scripts/construir_notebook.py --notebook entregable --ejecutar  # notebook entregable (02)
  python scripts/construir_notebook.py --notebook entregable --reusar-salidas
Opciones: --hasta NN (incluye solo las fuentes con prefijo <= NN).
"""
import argparse
import re
import sys
from pathlib import Path

import jupytext
import nbformat
from nbclient import NotebookClient

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOKS = {   # nombre -> (carpeta de fuentes, notebook de salida)
    "estudio": (RAIZ / "notebooks" / "fuente", RAIZ / "notebooks" / "01_Material_de_estudio_Saber11.ipynb"),
    "entregable": (RAIZ / "notebooks" / "fuente_entregable",
                   RAIZ / "notebooks" / "02_Entregable_Proyecto_Integrador_Saber11.ipynb"),
}
PAQUETES_COLAB = ["scikit-learn", "imbalanced-learn", "xgboost", "ydata-profiling"]
EXPLICACION_PREVIA = "¿Qué hacemos y por qué?"


def versiones_fijadas():
    """Lee requirements-notebook.txt y devuelve 'paquete==versión' para los paquetes que se instalan en Colab."""
    lineas = (RAIZ / "requirements-notebook.txt").read_text(encoding="utf-8").splitlines()
    fijadas = {l.split("==")[0].strip().lower(): l.strip() for l in lineas if "==" in l}
    # ydata-profiling importa pkg_resources, que setuptools >= 81 ya no incluye
    return [fijadas[p] for p in PAQUETES_COLAB] + ["setuptools<81"]


def indice_codigo_estudio():
    """Código de cada celda del notebook de estudio, indexado por su número de sección ('3.3', '5.2.1', ...)."""
    indice = {}
    for fuente in sorted(NOTEBOOKS["estudio"][0].glob("[0-9][0-9]_*.py")):
        for celda in jupytext.read(fuente).cells:
            m = re.match(r"# (\d+(?:\.\d+)*) ", celda.source) if celda.cell_type == "code" else None
            if m:
                indice[m.group(1)] = celda.source
    return indice


def expandir_copias(nb):
    """Reemplaza cada celda '#@copiar X como Y' por el código de la celda X del notebook de estudio, renumerada a Y."""
    indice = None
    for celda in nb.cells:
        m = re.fullmatch(r"#@copiar (\S+)(?: como (\S+))?", celda.source.strip()) if celda.cell_type == "code" else None
        if m:
            indice = indice or indice_codigo_estudio()
            origen, destino = m.group(1), m.group(2) or m.group(1)
            if origen not in indice:
                sys.exit(f"No existe la celda {origen} en el notebook de estudio")
            celda.source = re.sub(rf"^# {re.escape(origen)} ", f"# {destino} ", indice[origen], count=1)


def construir(dir_fuentes, hasta):
    fuentes = sorted(f for f in dir_fuentes.glob("[0-9][0-9]_*.py") if f.name[:2] <= hasta)
    celdas = []
    for fuente in fuentes:
        celdas.extend(jupytext.read(fuente).cells)
    nb = nbformat.v4.new_notebook(cells=celdas)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    expandir_copias(nb)
    for celda in nb.cells:
        celda.metadata = {}
        if celda.cell_type == "code":
            celda.source = celda.source.replace("{{PAQUETES}}", repr(versiones_fijadas()))
    return nb


def validar_estructura(nb, exigir_explicacion_previa):
    """Markdown debajo de cada celda de código; en el entregable también la explicación previa."""
    errores = []
    for i, celda in enumerate(nb.cells):
        if celda.cell_type != "code":
            continue
        if i + 1 >= len(nb.cells) or nb.cells[i + 1].cell_type != "markdown":
            errores.append(f"Celda de código #{i} sin Markdown debajo: {celda.source[:60]!r}")
        if exigir_explicacion_previa and (i == 0 or EXPLICACION_PREVIA not in nb.cells[i - 1].source):
            errores.append(f"Celda de código #{i} sin '{EXPLICACION_PREVIA}' antes: {celda.source[:60]!r}")
    return errores


def reusar_salidas(nb, salida):
    previo = nbformat.read(salida, as_version=4)
    nuevos = [c for c in nb.cells if c.cell_type == "code"]
    viejos = [c for c in previo.cells if c.cell_type == "code"]
    if [c.source for c in nuevos] != [c.source for c in viejos]:
        sys.exit("El código cambió respecto al notebook ejecutado: use --ejecutar")
    for nuevo, viejo in zip(nuevos, viejos):
        nuevo.outputs, nuevo.execution_count = viejo.outputs, viejo.execution_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", choices=list(NOTEBOOKS), default="estudio")
    parser.add_argument("--hasta", default="99", help="Prefijo máximo de fuente a incluir (p. ej. 03)")
    parser.add_argument("--ejecutar", action="store_true")
    parser.add_argument("--reusar-salidas", action="store_true")
    args = parser.parse_args()

    dir_fuentes, salida = NOTEBOOKS[args.notebook]
    nb = construir(dir_fuentes, args.hasta)
    errores = validar_estructura(nb, exigir_explicacion_previa=args.notebook == "entregable")
    if errores:
        sys.exit("\n".join(errores))
    if args.reusar_salidas:
        reusar_salidas(nb, salida)
    if args.ejecutar:
        NotebookClient(nb, timeout=None, kernel_name="python3",
                       resources={"metadata": {"path": str(salida.parent)}}).execute()
    nbformat.write(nb, salida)
    print(f"Notebook escrito: {salida.name} ({len(nb.cells)} celdas)")


if __name__ == "__main__":
    main()
