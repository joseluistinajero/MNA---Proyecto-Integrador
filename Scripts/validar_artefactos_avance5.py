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

    sha_actual = calcular_sha256(split_path)

    if sha_actual != SHA_DATASET_V2:
        raise RuntimeError(
            "El SHA 256 del dataset v2 no coincide "
            "con el registrado en el Avance 4"
        )

    df = pd.read_csv(split_path)
    elegibles = df[df["elegible_avance4"]].copy()

    conteos = (
        elegibles
        .groupby("split")
        .size()
        .to_dict()
    )

    esperados = {
        "train": 702,
        "val": 136,
        "test": 136
    }

    if conteos != esperados:
        raise RuntimeError(
            f"Distribución inesperada del dataset: {conteos}"
        )

    n_grupos = elegibles["grupo_experimental"].nunique()

    if n_grupos != 63:
        raise RuntimeError(
            f"Número inesperado de grupos experimentales: {n_grupos}"
        )

    checkpoints_requeridos = {
        "R(2+1)D-18/H5": checkpoint_dir / "r2plus1d_18_avance4_h5.pt",
        "MC3-18/H5": checkpoint_dir / "mc3_18_avance4_h5.pt"
    }

    mvit_path = checkpoint_dir / "mvit_v2_s_avance4_inicial.pt"

    resultados = []

    for modelo, path in checkpoints_requeridos.items():
        resultado = validar_checkpoint(path)
        resultado["modelo"] = modelo
        resultados.append(resultado)

    if mvit_path.is_file():
        try:
            resultado_mvit = validar_checkpoint(mvit_path)
            resultado_mvit["modelo"] = "MViT-V2-S"
            resultado_mvit["estado"] = "OK (histórico)"
            resultados.append(resultado_mvit)

        except (RuntimeError, OSError, EOFError) as exc:
            resultados.append({
                "modelo": "MViT-V2-S",
                "archivo": mvit_path.name,
                "epoch": None,
                "val_ap": None,
                "estado": f"Opcional no compatible: {type(exc).__name__}"
            })

    else:
        resultados.append({
            "modelo": "MViT-V2-S",
            "archivo": mvit_path.name,
            "epoch": None,
            "val_ap": None,
            "estado": "No disponible (opcional)"
        })

    tabla_artefactos = pd.DataFrame(resultados)[
        [
            "modelo",
            "archivo",
            "epoch",
            "val_ap",
            "estado"
        ]
    ]

    resumen_dataset = {
        "sha256": sha_actual,
        "n_registros": len(elegibles),
        "n_grupos": n_grupos,
        "particiones": conteos,
        "estado": "OK"
    }

    return tabla_artefactos, resumen_dataset
