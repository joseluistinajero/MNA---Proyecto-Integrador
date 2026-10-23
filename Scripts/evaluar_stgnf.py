from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import gaussian_filter1d

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    roc_curve
)

# ============================================================
# Configuración de la raíz del repositorio
# ============================================================

REPO_DIR = Path(__file__).resolve().parents[1]

REPO_DIR_STR = str(REPO_DIR)

if REPO_DIR_STR not in sys.path:
    sys.path.insert(
        0,
        REPO_DIR_STR
    )


from Datos.Preparacion_Datasets.scripts.dataset_retails_stgnf_npz import (
    RetailSSTGNFDatasetNPZ
)


def inferir_ventanas_stgnf(
    modelo,
    indice,
    particiones_cache,
    device,
    batch_size=256,
    cache_max_archivos=2,
    num_workers=0
):
    """
    Ejecuta inferencia STG-NF sobre un índice de ventanas.

    Devuelve un DataFrame que conserva una fila por ventana y agrega:
        - nll
        - score_anomalia

    La convención del proyecto utiliza NLL directamente como score
    de anomalía, por lo que valores mayores indican mayor anomalía.
    """

    if len(indice) == 0:
        raise ValueError(
            "El índice de evaluación no contiene ventanas."
        )

    indice_eval = (
        indice
        .reset_index(drop=True)
        .copy()
    )

    dataset = RetailSSTGNFDatasetNPZ(
        indice_eval,
        particiones_cache,
        cache_max_archivos=cache_max_archivos
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False
    )

    modelo.eval()

    scores_nll = []

    inicio = perf_counter()

    with torch.inference_mode():

        for batch_x, _ in loader:

            batch_x = batch_x.to(
                device,
                non_blocking=True
            )

            etiquetas_normales = torch.ones(
                batch_x.shape[0],
                device=device
            )

            _, nll = modelo(
                batch_x.float(),
                label=etiquetas_normales
            )

            if nll is None:
                raise RuntimeError(
                    "STG-NF devolvió NLL=None durante inferencia."
                )

            nll = (
                nll
                .reshape(-1)
                .detach()
                .cpu()
                .numpy()
            )

            scores_nll.append(nll)

    tiempo_s = perf_counter() - inicio

    scores_nll = np.concatenate(
        scores_nll
    )

    if len(scores_nll) != len(indice_eval):
        raise RuntimeError(
            "El número de scores no coincide con "
            "el número de ventanas evaluadas."
        )

    if not np.isfinite(scores_nll).all():
        raise RuntimeError(
            "La inferencia produjo scores NLL no finitos."
        )

    resultados = indice_eval.copy()

    resultados["nll"] = scores_nll

    resultados["score_anomalia"] = (
        resultados["nll"]
    )

    resumen = {
        "ventanas": int(len(resultados)),
        "batches": int(len(loader)),
        "batch_size": int(batch_size),
        "tiempo_s": float(tiempo_s),
        "throughput_ventanas_s": float(
            len(resultados) / tiempo_s
        ),
        "nll_min": float(scores_nll.min()),
        "nll_max": float(scores_nll.max()),
        "nll_mean": float(scores_nll.mean()),
        "nll_std": float(scores_nll.std()),
        "scores_finitos": bool(
            np.isfinite(scores_nll).all()
        )
    }

    return resultados, resumen


def validar_resultados_inferencia(
    resultados,
    resumen
):
    columnas_requeridas = {
        "frame_inicio",
        "frame_fin",
        "frame_score",
        "persona_id",
        "nll",
        "score_anomalia"
    }

    faltantes = (
        columnas_requeridas
        - set(resultados.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    if len(resultados) != resumen["ventanas"]:
        raise ValueError(
            "El resumen y el DataFrame contienen "
            "cantidades distintas de ventanas."
        )

    if not np.isfinite(
        resultados["score_anomalia"].to_numpy()
    ).all():
        raise ValueError(
            "Existen scores de anomalía no finitos."
        )

    return True

def agregar_scores_frame_stgnf(
    resultados_ventana
):
    """
    Agrega los scores STG-NF a nivel de frame.

    Cada score de ventana ya está asociado con su frame central
    mediante frame_score. Cuando varias personas aportan un score
    al mismo frame, se conserva el mayor NLL, equivalente a
    seleccionar la menor normalidad en la implementación oficial.
    """

    columnas_requeridas = {
        "particion",
        "ruta_json_relativa",
        "persona_id",
        "frame_score",
        "score_anomalia"
    }

    faltantes = (
        columnas_requeridas
        - set(resultados_ventana.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas para la agregación: "
            + ", ".join(
                sorted(faltantes)
            )
        )

    datos = (
        resultados_ventana[
            [
                "particion",
                "ruta_json_relativa",
                "persona_id",
                "frame_score",
                "score_anomalia"
            ]
        ]
        .copy()
    )

    duplicados_persona_frame = (
        datos
        .duplicated(
            subset=[
                "particion",
                "ruta_json_relativa",
                "persona_id",
                "frame_score"
            ]
        )
        .sum()
    )

    if duplicados_persona_frame != 0:
        raise ValueError(
            "Se encontraron múltiples scores para una misma "
            "persona y frame_score."
        )

    resultados_frame = (
        datos
        .groupby(
            [
                "particion",
                "ruta_json_relativa",
                "frame_score"
            ],
            as_index=False
        )
        .agg(
            score_anomalia=(
                "score_anomalia",
                "max"
            ),
            personas=(
                "persona_id",
                "nunique"
            ),
            ventanas=(
                "persona_id",
                "size"
            )
        )
        .sort_values(
            [
                "particion",
                "ruta_json_relativa",
                "frame_score"
            ]
        )
        .reset_index(drop=True)
    )

    if not np.isfinite(
        resultados_frame["score_anomalia"]
        .to_numpy()
    ).all():
        raise ValueError(
            "La agregación frame-level produjo "
            "scores no finitos."
        )

    resumen = {
        "ventanas_entrada":
            int(len(datos)),

        "frames_con_score":
            int(len(resultados_frame)),

        "fuentes":
            int(
                resultados_frame[
                    "ruta_json_relativa"
                ].nunique()
            ),

        "personas_max_por_frame":
            int(
                resultados_frame[
                    "personas"
                ].max()
            ),

        "frames_multipersona":
            int(
                (
                    resultados_frame[
                        "personas"
                    ]
                    > 1
                ).sum()
            ),

        "duplicados_persona_frame":
            int(
                duplicados_persona_frame
            ),

        "scores_finitos":
            bool(
                np.isfinite(
                    resultados_frame[
                        "score_anomalia"
                    ].to_numpy()
                ).all()
            )
    }

    return resultados_frame, resumen

def alinear_ground_truth_frame_stgnf(
    resultados_frame,
    raiz_particion
):
    """
    Alinea scores frame-level de STG-NF con las máscaras binarias
    de RetailS.

    Se espera la estructura:

        raiz_particion/
            pose/test/
            gt/test_frame_mask/

    Para cada ruta_json_relativa, la máscara correspondiente tiene
    el mismo nombre base y extensión .npy.

    La etiqueta de cada observación se obtiene mediante:

        etiqueta = mascara[frame_score]
    """

    raiz_particion = Path(raiz_particion)

    directorio_gt = (
        raiz_particion
        / "gt"
        / "test_frame_mask"
    )

    if not directorio_gt.is_dir():
        raise FileNotFoundError(
            "No se encontró el directorio de ground truth "
            "esperado para la partición."
        )

    columnas_requeridas = {
        "particion",
        "ruta_json_relativa",
        "frame_score",
        "score_anomalia"
    }

    faltantes = (
        columnas_requeridas
        - set(resultados_frame.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas para alinear "
            "el ground truth: "
            + ", ".join(sorted(faltantes))
        )

    datos = resultados_frame.copy()

    etiquetas = np.empty(
        len(datos),
        dtype=np.uint8
    )

    mascaras_cache = {}

    fuentes_sin_gt = []
    fuentes_gt_no_binario = []
    frames_fuera_rango = []

    for i, fila in enumerate(
        datos.itertuples(index=False)
    ):
        ruta_json = Path(
            fila.ruta_json_relativa
        )

        stem = ruta_json.stem

        if stem not in mascaras_cache:
            ruta_gt = (
                directorio_gt
                / f"{stem}.npy"
            )

            if not ruta_gt.is_file():
                fuentes_sin_gt.append(stem)
                continue

            mascara = np.load(
                ruta_gt,
                allow_pickle=False
            )

            mascara = np.asarray(
                mascara
            ).reshape(-1)

            valores = set(
                np.unique(mascara).tolist()
            )

            if not valores.issubset({0, 1}):
                fuentes_gt_no_binario.append(
                    stem
                )
                continue

            mascaras_cache[stem] = mascara

        if stem not in mascaras_cache:
            continue

        mascara = mascaras_cache[stem]

        frame_score = int(
            fila.frame_score
        )

        if (
            frame_score < 0
            or frame_score >= len(mascara)
        ):
            frames_fuera_rango.append(
                (
                    stem,
                    frame_score,
                    len(mascara)
                )
            )
            continue

        etiquetas[i] = mascara[
            frame_score
        ]

    if fuentes_sin_gt:
        raise ValueError(
            "Existen fuentes sin ground truth. "
            f"Cantidad: {len(set(fuentes_sin_gt))}."
        )

    if fuentes_gt_no_binario:
        raise ValueError(
            "Existen máscaras con valores distintos de 0 y 1. "
            f"Cantidad: {len(set(fuentes_gt_no_binario))}."
        )

    if frames_fuera_rango:
        raise ValueError(
            "Existen frame_score fuera de los límites de "
            "su máscara de ground truth. "
            f"Cantidad: {len(frames_fuera_rango)}."
        )

    datos["etiqueta"] = etiquetas

    resumen = {
        "frames_alineados":
            int(len(datos)),

        "fuentes_evaluadas":
            int(len(mascaras_cache)),

        "frames_normales":
            int(
                (datos["etiqueta"] == 0).sum()
            ),

        "frames_anomalos":
            int(
                (datos["etiqueta"] == 1).sum()
            ),

        "prevalencia_anomalia":
            float(
                datos["etiqueta"].mean()
            ),

        "fuentes_sin_gt": 0,

        "gt_no_binario": 0,

        "frames_fuera_rango": 0
    }

    return datos, resumen

def construir_cobertura_completa_stgnf(
    resultados_frame,
    raiz_particion
):
    """
    Reconstruye la evaluación frame-level completa de RetailS.

    Incluye todos los frames de todas las máscaras de ground truth.
    Los frames sin score STG-NF reciben el menor NLL finito global,
    equivalente al relleno con máxima normalidad utilizado por la
    implementación oficial.
    """

    columnas_requeridas = {
        "particion",
        "ruta_json_relativa",
        "frame_score",
        "score_anomalia"
    }

    faltantes = (
        columnas_requeridas
        - set(resultados_frame.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas: "
            + ", ".join(sorted(faltantes))
        )

    datos = resultados_frame.copy()

    if len(datos) == 0:
        raise ValueError(
            "No existen scores frame-level para reconstruir "
            "la cobertura completa."
        )

    if not np.isfinite(
        datos["score_anomalia"].to_numpy()
    ).all():
        raise ValueError(
            "Los scores observados contienen valores no finitos."
        )

    particiones = (
        datos["particion"]
        .dropna()
        .unique()
    )

    if len(particiones) != 1:
        raise ValueError(
            "La reconstrucción debe realizarse sobre "
            "una sola partición."
        )

    particion = particiones[0]

    datos["fuente"] = (
        datos["ruta_json_relativa"]
        .map(lambda ruta: Path(ruta).stem)
    )

    duplicados = (
        datos
        .duplicated(
            subset=[
                "fuente",
                "frame_score"
            ]
        )
        .sum()
    )

    if duplicados != 0:
        raise ValueError(
            "Existen múltiples scores agregados para "
            "la misma fuente y frame."
        )

    raiz_particion = Path(
        raiz_particion
    )

    directorio_gt = (
        raiz_particion
        / "gt"
        / "test_frame_mask"
    )

    if not directorio_gt.is_dir():
        raise FileNotFoundError(
            "No se encontró el directorio de ground truth."
        )

    archivos_gt = sorted(
        directorio_gt.glob("*.npy")
    )

    if not archivos_gt:
        raise FileNotFoundError(
            "No se encontraron máscaras .npy de ground truth."
        )

    bloques = []
    longitudes_gt = {}

    for ruta_gt in archivos_gt:
        fuente = ruta_gt.stem

        mascara = np.load(
            ruta_gt,
            allow_pickle=False
        )

        mascara = np.asarray(
            mascara
        ).reshape(-1)

        valores = set(
            np.unique(mascara).tolist()
        )

        if not valores.issubset({0, 1}):
            raise ValueError(
                f"La máscara {ruta_gt.name} "
                "contiene valores distintos de 0 y 1."
            )

        longitudes_gt[fuente] = len(
            mascara
        )

        bloques.append(
            pd.DataFrame(
                {
                    "particion": particion,
                    "fuente": fuente,
                    "frame_score": np.arange(
                        len(mascara),
                        dtype=np.int64
                    ),
                    "etiqueta": mascara.astype(
                        np.uint8
                    )
                }
            )
        )

    cobertura = pd.concat(
        bloques,
        ignore_index=True
    )

    fuentes_gt = set(
        longitudes_gt
    )

    fuentes_scores = set(
        datos["fuente"].unique()
    )

    fuentes_sin_gt = (
        fuentes_scores - fuentes_gt
    )

    if fuentes_sin_gt:
        raise ValueError(
            "Existen fuentes con score pero sin ground truth: "
            + ", ".join(
                sorted(fuentes_sin_gt)[:10]
            )
        )

    fuera_rango = datos[
        datos.apply(
            lambda fila:
                int(fila["frame_score"]) < 0
                or int(fila["frame_score"])
                >= longitudes_gt[
                    fila["fuente"]
                ],
            axis=1
        )
    ]

    if len(fuera_rango) != 0:
        raise ValueError(
            "Existen scores asociados con frames fuera "
            "del rango de su ground truth."
        )

    observados = (
        datos[
            [
                "fuente",
                "frame_score",
                "score_anomalia"
            ]
        ]
        .rename(
            columns={
                "score_anomalia":
                    "score_anomalia_observado"
            }
        )
    )

    cobertura = cobertura.merge(
        observados,
        on=[
            "fuente",
            "frame_score"
        ],
        how="left",
        validate="one_to_one"
    )

    cobertura["score_observado"] = (
        cobertura[
            "score_anomalia_observado"
        ].notna()
    )

    score_relleno = float(
        datos["score_anomalia"].min()
    )

    cobertura[
        "score_anomalia_sin_suavizar"
    ] = (
        cobertura[
            "score_anomalia_observado"
        ]
        .fillna(score_relleno)
    )

    if not np.isfinite(
        cobertura[
            "score_anomalia_sin_suavizar"
        ].to_numpy()
    ).all():
        raise RuntimeError(
            "La cobertura completa contiene scores "
            "no finitos después del relleno."
        )

    fuentes_con_score = int(
        cobertura.loc[
            cobertura["score_observado"],
            "fuente"
        ].nunique()
    )

    resumen = {
        "fuentes_gt":
            int(len(archivos_gt)),

        "fuentes_con_score":
            fuentes_con_score,

        "fuentes_sin_score":
            int(
                len(archivos_gt)
                - fuentes_con_score
            ),

        "frames_totales":
            int(len(cobertura)),

        "frames_con_score_observado":
            int(
                cobertura[
                    "score_observado"
                ].sum()
            ),

        "frames_rellenados":
            int(
                (
                    ~cobertura[
                        "score_observado"
                    ]
                ).sum()
            ),

        "score_relleno_nll":
            score_relleno,

        "frames_normales":
            int(
                (
                    cobertura["etiqueta"] == 0
                ).sum()
            ),

        "frames_anomalos":
            int(
                (
                    cobertura["etiqueta"] == 1
                ).sum()
            ),

        "prevalencia_anomalia":
            float(
                cobertura[
                    "etiqueta"
                ].mean()
            )
    }

    return cobertura, resumen

def aplicar_suavizado_temporal_stgnf(
    cobertura,
    sigmas=(1, 2, 3, 4, 5, 6)
):
    """
    Aplica el suavizado temporal utilizado por STG-NF.

    El filtro gaussiano se aplica sucesivamente a cada fuente,
    preservando sus límites temporales.

    Requiere la cobertura frame-level completa generada por
    construir_cobertura_completa_stgnf().
    """

    columnas_requeridas = {
        "fuente",
        "frame_score",
        "score_anomalia_sin_suavizar"
    }

    faltantes = (
        columnas_requeridas
        - set(cobertura.columns)
    )

    if faltantes:
        raise ValueError(
            "Faltan columnas requeridas para suavizado: "
            + ", ".join(sorted(faltantes))
        )

    if not sigmas:
        raise ValueError(
            "Debe especificarse al menos un valor de sigma."
        )

    sigmas = tuple(
        float(sigma)
        for sigma in sigmas
    )

    if any(
        sigma <= 0
        for sigma in sigmas
    ):
        raise ValueError(
            "Todos los valores de sigma deben ser positivos."
        )

    resultados = cobertura.copy()

    if resultados.duplicated(
        subset=[
            "fuente",
            "frame_score"
        ]
    ).any():
        raise ValueError(
            "Existen frames duplicados dentro de una fuente."
        )

    if not np.isfinite(
        resultados[
            "score_anomalia_sin_suavizar"
        ].to_numpy()
    ).all():
        raise ValueError(
            "Existen scores no finitos antes del suavizado."
        )

    resultados[
        "score_anomalia_suavizado"
    ] = np.nan

    fuentes_procesadas = 0

    for fuente, indices in resultados.groupby(
        "fuente",
        sort=False
    ).groups.items():

        indices = np.asarray(
            list(indices),
            dtype=np.int64
        )

        bloque = (
            resultados
            .loc[
                indices,
                [
                    "frame_score",
                    "score_anomalia_sin_suavizar"
                ]
            ]
            .sort_values("frame_score")
        )

        frames = (
            bloque["frame_score"]
            .to_numpy(dtype=np.int64)
        )

        esperado = np.arange(
            len(frames),
            dtype=np.int64
        )

        if not np.array_equal(
            frames,
            esperado
        ):
            raise ValueError(
                f"La fuente {fuente} no contiene una "
                "secuencia completa de frames 0..N-1."
            )

        scores = (
            bloque[
                "score_anomalia_sin_suavizar"
            ]
            .to_numpy(dtype=np.float64)
            .copy()
        )

        for sigma in sigmas:
            scores = gaussian_filter1d(
                scores,
                sigma=sigma
            )

        resultados.loc[
            bloque.index,
            "score_anomalia_suavizado"
        ] = scores

        fuentes_procesadas += 1

    if resultados[
        "score_anomalia_suavizado"
    ].isna().any():
        raise RuntimeError(
            "Quedaron frames sin score después del suavizado."
        )

    if not np.isfinite(
        resultados[
            "score_anomalia_suavizado"
        ].to_numpy()
    ).all():
        raise RuntimeError(
            "El suavizado produjo scores no finitos."
        )

    diferencia = (
        resultados[
            "score_anomalia_suavizado"
        ]
        - resultados[
            "score_anomalia_sin_suavizar"
        ]
    )

    resumen = {
        "fuentes_procesadas":
            int(fuentes_procesadas),

        "frames_procesados":
            int(len(resultados)),

        "sigmas":
            ",".join(
                str(int(sigma))
                if sigma.is_integer()
                else str(sigma)
                for sigma in sigmas
            ),

        "score_raw_min":
            float(
                resultados[
                    "score_anomalia_sin_suavizar"
                ].min()
            ),

        "score_raw_max":
            float(
                resultados[
                    "score_anomalia_sin_suavizar"
                ].max()
            ),

        "score_smooth_min":
            float(
                resultados[
                    "score_anomalia_suavizado"
                ].min()
            ),

        "score_smooth_max":
            float(
                resultados[
                    "score_anomalia_suavizado"
                ].max()
            ),

        "diferencia_absoluta_media":
            float(
                diferencia.abs().mean()
            ),

        "diferencia_absoluta_maxima":
            float(
                diferencia.abs().max()
            ),

        "scores_finitos":
            True
    }

    return resultados, resumen

def calcular_metricas_ranking_stgnf(
    y_true,
    scores
):
    """
    Calcula métricas independientes de umbral para STG-NF.

    Utiliza las mismas definiciones empleadas en el notebook
    para los modelos RGB:
        - Average Precision
        - PR-AUC
        - ROC-AUC
        - Equal Error Rate
    """

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    scores = np.asarray(
        scores,
        dtype=np.float64
    ).reshape(-1)

    if len(y_true) != len(scores):
        raise ValueError(
            "Las etiquetas y los scores tienen "
            "longitudes diferentes."
        )

    if len(y_true) == 0:
        raise ValueError(
            "No existen observaciones para evaluar."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Los scores contienen valores no finitos."
        )

    clases = set(
        np.unique(y_true).tolist()
    )

    if not clases.issubset({0, 1}):
        raise ValueError(
            "Las etiquetas deben contener únicamente 0 y 1."
        )

    if len(clases) != 2:
        raise ValueError(
            "Se requieren ambas clases para calcular "
            "las métricas."
        )

    average_precision = average_precision_score(
        y_true,
        scores
    )

    precision_curve, recall_curve, _ = (
        precision_recall_curve(
            y_true,
            scores
        )
    )

    pr_auc = auc(
        recall_curve,
        precision_curve
    )

    roc_auc = roc_auc_score(
        y_true,
        scores
    )

    fpr, tpr, thresholds = roc_curve(
    y_true,
    scores
    )

    fnr = 1 - tpr

    idx_eer = np.nanargmin(
        np.abs(
            fpr - fnr
        )
    )

    eer = (
        fpr[idx_eer]
        + fnr[idx_eer]
    ) / 2

    threshold_eer = float(
        thresholds[idx_eer]
    )

    return {
        "average_precision":
            float(average_precision),

        "pr_auc":
            float(pr_auc),

        "roc_auc":
            float(roc_auc),

        "eer":
            float(eer),
        "threshold_eer":
            threshold_eer,
        "observaciones":
            int(len(y_true)),

        "positivos":
            int(y_true.sum()),

        "negativos":
            int(
                len(y_true)
                - y_true.sum()
            )
    }

def calcular_metricas_umbral_stgnf(
    y_true,
    scores,
    threshold
):
    """
    Calcula métricas de clasificación para STG-NF
    usando un umbral previamente fijado.
    """

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    scores = np.asarray(
        scores,
        dtype=np.float64
    ).reshape(-1)

    if len(y_true) != len(scores):
        raise ValueError(
            "Las etiquetas y los scores tienen "
            "longitudes diferentes."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Los scores contienen valores no finitos."
        )

    y_pred = (
        scores >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else np.nan
    )

    return {
        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    y_true,
                    y_pred
                )
            ),

        "precision":
            float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "recall":
            float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "specificity":
            float(specificity),

        "f1":
            float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0
                )
            ),

        "tn":
            int(tn),

        "fp":
            int(fp),

        "fn":
            int(fn),

        "tp":
            int(tp)
    }

def main():
    print(
        "Módulo de evaluación STG-NF disponible."
    )
    print(
        "Funciones: inferir_ventanas_stgnf, "
        "validar_resultados_inferencia, "
        "agregar_scores_frame_stgnf, "
        "alinear_ground_truth_frame_stgnf, "
        "construir_cobertura_completa_stgnf, "
        "aplicar_suavizado_temporal_stgnf, "
        "calcular_metricas_ranking_stgnf",
        "calcular_metricas_umbral_stgnf"
    )
if __name__ == "__main__":
    main()