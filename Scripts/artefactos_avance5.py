from pathlib import Path

import numpy as np
import pandas as pd


def guardar_embeddings_npz(
    path,
    embeddings,
    clip_id,
    grupo_experimental,
    split,
    label
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings),
        clip_id=np.asarray(clip_id),
        grupo_experimental=np.asarray(grupo_experimental),
        split=np.asarray(split),
        label=np.asarray(label)
    )


def cargar_embeddings_npz(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de embeddings: {path.name}"
        )

    with np.load(path, allow_pickle=False) as datos:
        return {clave: datos[clave] for clave in datos.files}


def validar_embeddings_npz(
    path,
    split_esperado,
    n_esperado,
    dim_esperada=512
):
    datos = cargar_embeddings_npz(path)

    claves_requeridas = {
        "embeddings",
        "clip_id",
        "grupo_experimental",
        "split",
        "label"
    }

    faltantes = claves_requeridas - set(datos)

    if faltantes:
        raise RuntimeError(
            f"Faltan campos en {Path(path).name}: {sorted(faltantes)}"
        )

    embeddings = datos["embeddings"]
    n = len(datos["clip_id"])

    if embeddings.shape != (n_esperado, dim_esperada):
        raise RuntimeError(
            f"Dimensión inesperada de embeddings: {embeddings.shape}"
        )

    if n != n_esperado:
        raise RuntimeError(
            f"Número inesperado de clips: {n}"
        )

    for clave in ["grupo_experimental", "split", "label"]:
        if len(datos[clave]) != n:
            raise RuntimeError(
                f"Longitud incompatible en el campo {clave}"
            )

    if len(np.unique(datos["clip_id"])) != n:
        raise RuntimeError(
            "Se encontraron clip_id duplicados"
        )

    if not np.all(datos["split"] == split_esperado):
        raise RuntimeError(
            f"El artefacto contiene registros fuera de {split_esperado}"
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError(
            "Los embeddings contienen valores no finitos"
        )

    if not np.isin(datos["label"], [0, 1]).all():
        raise RuntimeError(
            "Se encontraron etiquetas fuera de la clasificación binaria"
        )

    return {
        "archivo": Path(path).name,
        "n_clips": n,
        "dim_embedding": embeddings.shape[1],
        "split": split_esperado,
        "estado": "OK"
    }


def guardar_probabilidades_csv(path, df):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def cargar_probabilidades_csv(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo de probabilidades: {path.name}"
        )

    return pd.read_csv(path)


def validar_probabilidades_csv(
    path,
    columnas_modelos,
    split_esperado,
    n_esperado
):
    df = cargar_probabilidades_csv(path)

    columnas_requeridas = {
        "clip_id",
        "grupo_experimental",
        "split",
        "label",
        *columnas_modelos
    }

    faltantes = columnas_requeridas - set(df.columns)

    if faltantes:
        raise RuntimeError(
            f"Faltan columnas en {Path(path).name}: {sorted(faltantes)}"
        )

    if len(df) != n_esperado:
        raise RuntimeError(
            f"Número inesperado de clips: {len(df)}"
        )

    if df["clip_id"].duplicated().any():
        raise RuntimeError(
            "Se encontraron clip_id duplicados"
        )

    if not df["split"].eq(split_esperado).all():
        raise RuntimeError(
            f"El artefacto contiene registros fuera de {split_esperado}"
        )

    if not df["label"].isin([0, 1]).all():
        raise RuntimeError(
            "Se encontraron etiquetas fuera de la clasificación binaria"
        )

    probabilidades = df[columnas_modelos].to_numpy()

    if not np.isfinite(probabilidades).all():
        raise RuntimeError(
            "Las probabilidades contienen valores no finitos"
        )

    if ((probabilidades < 0) | (probabilidades > 1)).any():
        raise RuntimeError(
            "Se encontraron probabilidades fuera del intervalo [0, 1]"
        )

    return {
        "archivo": Path(path).name,
        "n_clips": len(df),
        "split": split_esperado,
        "modelos": len(columnas_modelos),
        "estado": "OK"
    }