"""Pruebas de la app Streamlit con el framework AppTest (sin navegador)."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_carga_sin_errores():
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    assert "Saber 11" in at.title[0].value
    assert any("fines exclusivamente educativos" in aviso.value for aviso in at.info)   # Aviso académico visible


def test_prediccion_individual_muestra_resultado():
    at = AppTest.from_file(APP, default_timeout=120).run()
    next(b for b in at.button if b.label == "Predecir").click().run()
    assert not at.exception
    assert any("Probabilidad" in m.label or "Puntaje" in m.label for m in at.metric)
    assert any("Banda de acompañamiento" in md.value for md in at.markdown)   # Capa prescriptiva visible


def test_prediccion_por_lotes_con_datos_de_ejemplo():
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.button(key="btn_ejemplo").click().run()
    assert not at.exception
    assert at.session_state["n_lote"] == 1000
    assert any("Bandas de acompañamiento" in md.value for md in at.markdown)  # Resumen por banda del lote
