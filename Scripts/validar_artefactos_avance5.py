from pathlib import Path
import hashlib

import pandas as pd
import torch


SHA_DATASET_V2 = (
    "b85d6b55e8c32420a4df5a0fb9a10e558105"
    "beee47f1d4a98ad83d63fcbda21e"
)

PARTICIONES_ESPERADAS = {
    "train": 702,
    "val": 136,
    "test": 136
}

CLAVES_CHECKPOINT = {
    "model_state_dict",
    "experiment_config",
    "val_average_precision"
}

COLUMNAS_HISTORIAL = {
    "epoch",
    "val_average_precision"
}


def calcular_sha256(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        for bloque in iter(lambda: f.read(8192), b""):
            h.update(bloque)

    return h.hexdigest()


def validar_checkpoint(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Falta checkpoint requerido: {path.name}"
        )

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    faltantes = (
        CLAVES_CHECKPOINT
        - set(checkpoint.keys())
    )

    if faltantes:
        raise RuntimeError(
            f"Checkpoint incompatible {path.name}. "
            f"Faltan claves: {sorted(faltantes)}"
        )

    config = checkpoint["experiment_config"]

    if (
        config.get("dataset_v2_sha256")
        != SHA_DATASET_V2
    ):
        raise RuntimeError(
            f"El checkpoint {path.name} no corresponde "
            "al dataset v2 esperado"
        )

    return {
        "archivo": path.name,
        "epoch": checkpoint.get("epoch"),
        "val_ap": checkpoint.get(
            "val_average_precision"
        ),
        "estado": "OK"
    }


def validar_historial(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Falta historial requerido: {path.name}"
        )

    historial = pd.read_csv(path)

    if historial.empty:
        raise RuntimeError(
            f"El historial está vacío: {path.name}"
        )

    faltantes = (
        COLUMNAS_HISTORIAL
        - set(historial.columns)
    )

    if faltantes:
        raise RuntimeError(
            f"Historial incompatible {path.name}. "
            f"Faltan columnas: {sorted(faltantes)}"
        )

    return {
        "archivo": path.name,
        "epocas": len(historial),
        "estado": "OK"
    }


def validar_avance4(repo_dir):
    repo_dir = Path(repo_dir)

    split_path = (
        repo_dir
        / "Datos"
        / "Preparacion_Datasets"
        / "avance4_split_ground_truth_v2.csv"
    )

    checkpoint_dir = (
        repo_dir
        / "Notebooks"
        / "checkpoints"
    )

    if not split_path.is_file():
        raise FileNotFoundError(
            "No se encontró el dataset v2 requerido"
        )

    sha_actual = calcular_sha256(split_path)

    if sha_actual != SHA_DATASET_V2:
        raise RuntimeError(
            "El SHA 256 del dataset v2 no coincide "
            "con el registrado en el Avance 4"
        )

    df = pd.read_csv(split_path)

    columnas_requeridas = {
        "clip_id",
        "split",
        "grupo_experimental",
        "elegible_avance4"
    }

    faltantes = (
        columnas_requeridas
        - set(df.columns)
    )

    if faltantes:
        raise RuntimeError(
            "Faltan columnas requeridas en el dataset: "
            f"{sorted(faltantes)}"
        )

    elegibles = (
        df[df["elegible_avance4"]]
        .copy()
    )

    conteos = (
        elegibles["split"]
        .value_counts()
        .to_dict()
    )

    if conteos != PARTICIONES_ESPERADAS:
        raise RuntimeError(
            "Distribución inesperada del dataset: "
            f"{conteos}"
        )

    grupos_por_split = (
        elegibles
        .groupby("grupo_experimental")["split"]
        .nunique()
    )

    if (grupos_por_split > 1).any():
        raise RuntimeError(
            "Se detectaron grupos experimentales "
            "compartidos entre particiones"
        )

    modelos = {
        "R(2+1)D-18/H5": {
            "checkpoint":
                "r2plus1d_18_avance4_h5.pt",
            "historial":
                "r2plus1d_18_avance4_h5_history.csv"
        },
        "MC3-18/H5": {
            "checkpoint":
                "mc3_18_avance4_h5.pt",
            "historial":
                "mc3_18_avance4_h5_history.csv"
        },
        "MViT-V2-S": {
            "checkpoint":
                "mvit_v2_s_avance4_inicial.pt",
            "historial":
                "mvit_v2_s_avance4_inicial_history.csv"
        }
    }

    resultados = []

    for modelo, archivos in modelos.items():
        checkpoint = validar_checkpoint(
            checkpoint_dir
            / archivos["checkpoint"]
        )

        historial = validar_historial(
            checkpoint_dir
            / archivos["historial"]
        )

        resultados.append(
            {
                "modelo": modelo,
                "checkpoint": checkpoint["archivo"],
                "historial": historial["archivo"],
                "epoch": checkpoint["epoch"],
                "val_ap": checkpoint["val_ap"],
                "epocas_historial":
                    historial["epocas"],
                "estado": "OK"
            }
        )

    resumen_dataset = {
        "sha256": sha_actual,
        "n_registros": len(elegibles),
        "n_grupos":
            elegibles[
                "grupo_experimental"
            ].nunique(),
        "particiones": conteos,
        "estado": "OK"
    }

    tabla_modelos = pd.DataFrame(resultados)

    return tabla_modelos, resumen_dataset