"""Pruebas del pipeline serializado y su metadata (lo que consume la app)."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

RAIZ = Path(__file__).resolve().parents[1]
MODELO = RAIZ / "models" / "pipeline_saber11.joblib"
METADATA = RAIZ / "models" / "metadata.json"
NO_VISTOS = RAIZ / "data" / "nuevos" / "registros_no_vistos_1000.csv"


@pytest.fixture(scope="module")
def meta():
    return json.loads(METADATA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def modelo():
    # joblib usa pickle: solo se carga el artefacto generado por este mismo proyecto (fuente confiable)
    return joblib.load(MODELO)


def perfil_valido(meta):
    perfil = {var: meta["categorias"][var][0] for var in meta["features"] if var in meta["categorias"]}
    perfil["edad"] = 17
    return pd.DataFrame([perfil])[meta["features"]]


def puntaje(modelo, X):
    return modelo.predict_proba(X)[:, 1] if hasattr(modelo, "predict_proba") else modelo.decision_function(X)


def test_metadata_completa(meta):
    for clave in ["features", "categorias", "etiquetas", "umbral_decision", "tipo_puntaje", "modelo_final",
                  "metricas_test", "versiones", "rango_edad", "requiere_xgboost", "proposito", "valores_por_defecto"]:
        assert clave in meta, clave
    assert len(meta["features"]) == 19
    assert set(meta["etiquetas"]) >= set(meta["features"])
    # El perfil por defecto del formulario debe usar solo categorías válidas
    for var, valor in meta["valores_por_defecto"].items():
        assert valor in meta["categorias"][var], (var, valor)


def test_modelo_pesa_menos_de_95_mb():
    assert MODELO.stat().st_size < 95e6


def test_prediccion_de_un_perfil_valido(modelo, meta):
    s = puntaje(modelo, perfil_valido(meta))
    assert s.shape == (1,)
    if meta["tipo_puntaje"] == "probabilidad":
        assert 0.0 <= s[0] <= 1.0


def test_categoria_desconocida_no_rompe(modelo, meta):
    X = perfil_valido(meta)
    X.loc[0, "cole_jornada"] = "CATEGORIA_INEXISTENTE"
    X.loc[0, "fami_estratovivienda"] = "Estrato 9"
    assert np.isfinite(puntaje(modelo, X)).all()


def test_valores_faltantes_no_rompen(modelo, meta):
    X = perfil_valido(meta)
    X.loc[0, "edad"] = np.nan
    X.loc[0, "fami_educacionmadre"] = np.nan
    X.loc[0, "cole_bilingue"] = np.nan
    assert np.isfinite(puntaje(modelo, X)).all()


def test_desempeno_en_registros_no_vistos(modelo, meta):
    df = pd.read_csv(NO_VISTOS)
    prediccion = (puntaje(modelo, df[meta["features"]]) >= meta["umbral_decision"]).astype(int)
    assert f1_score(df["alto_desempeno"], prediccion) > 0.35
