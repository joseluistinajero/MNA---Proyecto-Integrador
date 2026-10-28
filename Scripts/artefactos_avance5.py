from pathlib import Path

import numpy as np
import pandas as pd
import json
import joblib

from sklearn.ensemble import AdaBoostClassifier

import torch

from Scripts.validar_artefactos_avance5 import calcular_sha256

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
        embeddings=np.asarray(embeddings, dtype=np.float32),
        clip_id=np.asarray(clip_id, dtype=str),
        grupo_experimental=np.asarray(grupo_experimental, dtype=str),
        split=np.asarray(split, dtype=str),
        label=np.asarray(label, dtype=np.int64)
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

def guardar_random_forest(
    modelo_path,
    busqueda_path,
    resumen_path,
    modelo,
    resultados_cv,
    resumen
):
    modelo_path = Path(modelo_path)
    busqueda_path = Path(busqueda_path)
    resumen_path = Path(resumen_path)

    modelo_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(modelo, modelo_path)
    pd.DataFrame(resultados_cv).to_csv(busqueda_path, index=False)

    with open(resumen_path, "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=4, ensure_ascii=False)


def cargar_random_forest(modelo_path, busqueda_path, resumen_path):
    modelo_path = Path(modelo_path)
    busqueda_path = Path(busqueda_path)
    resumen_path = Path(resumen_path)

    modelo = joblib.load(modelo_path)
    resultados_cv = pd.read_csv(busqueda_path)

    with open(resumen_path, "r", encoding="utf-8") as archivo:
        resumen = json.load(archivo)

    return modelo, resultados_cv, resumen


def validar_random_forest(
    modelo_path,
    busqueda_path,
    resumen_path,
    n_train_esperado,
    dim_embedding_esperada,
    n_iter_esperado,
    seed_esperada
):
    modelo_path = Path(modelo_path)
    busqueda_path = Path(busqueda_path)
    resumen_path = Path(resumen_path)

    for path in [modelo_path, busqueda_path, resumen_path]:
        if not path.exists():
            raise FileNotFoundError(f"No existe el artefacto requerido: {path.name}")

    modelo, resultados_cv, resumen = cargar_random_forest(
        modelo_path,
        busqueda_path,
        resumen_path
    )

    claves_requeridas = {
        "modelo",
        "extractor",
        "scoring",
        "cv",
        "n_splits",
        "n_iter",
        "seed",
        "class_weight",
        "best_score_cv",
        "best_params",
        "tiempo_busqueda_s",
        "tiempo_refit_s",
        "n_train",
        "dim_embedding",
        "dataset_v2_sha256"
    }

    faltantes = claves_requeridas - set(resumen)

    if faltantes:
        raise RuntimeError(
            f"El resumen de Random Forest no contiene las claves: {sorted(faltantes)}"
        )

    if resumen["modelo"] != "RandomForestClassifier":
        raise RuntimeError("El artefacto no corresponde a RandomForestClassifier.")

    if resumen["extractor"] != "r2plus1d_18_kinetics400":
        raise RuntimeError("El extractor registrado no corresponde a Kinetics-400.")

    if resumen["scoring"] != "average_precision":
        raise RuntimeError("La métrica registrada no corresponde a Average Precision.")

    if resumen["cv"] != "StratifiedGroupKFold":
        raise RuntimeError("La estrategia CV registrada no corresponde a StratifiedGroupKFold.")

    if resumen["n_train"] != n_train_esperado:
        raise RuntimeError("El número de observaciones de entrenamiento no coincide.")

    if resumen["dim_embedding"] != dim_embedding_esperada:
        raise RuntimeError("La dimensión de los embeddings no coincide.")

    if resumen["n_iter"] != n_iter_esperado:
        raise RuntimeError("El número de configuraciones de búsqueda no coincide.")

    if resumen["seed"] != seed_esperada:
        raise RuntimeError("La semilla experimental no coincide.")

    if len(resultados_cv) != n_iter_esperado:
        raise RuntimeError("El archivo de resultados CV no contiene el número esperado de configuraciones.")

    if not hasattr(modelo, "n_features_in_"):
        raise RuntimeError("El modelo almacenado no contiene información sobre sus variables de entrada.")

    if modelo.n_features_in_ != dim_embedding_esperada:
        raise RuntimeError("El modelo fue entrenado con una dimensión de entrada diferente.")

    return {
        "modelo": resumen["modelo"],
        "configuraciones": len(resultados_cv),
        "best_score_cv": resumen["best_score_cv"],
        "dim_embedding": resumen["dim_embedding"],
        "estado": "OK"
    }

def guardar_adaboost(
    modelo_path,
    busqueda_path,
    resumen_path,
    modelo,
    resultados_cv,
    resumen
):
    joblib.dump(modelo, modelo_path)
    pd.DataFrame(resultados_cv).to_csv(busqueda_path, index=False)

    with open(resumen_path, "w", encoding="utf-8") as archivo:
        json.dump(resumen, archivo, indent=2, ensure_ascii=False)


def cargar_adaboost(
    modelo_path,
    busqueda_path,
    resumen_path
):
    modelo = joblib.load(modelo_path)
    resultados_cv = pd.read_csv(busqueda_path)

    with open(resumen_path, "r", encoding="utf-8") as archivo:
        resumen = json.load(archivo)

    return modelo, resultados_cv, resumen


def validar_adaboost(
    modelo_path,
    busqueda_path,
    resumen_path,
    n_train_esperado,
    dim_embedding_esperada,
    n_iter_esperado,
    seed_esperada,
    dataset_sha256_esperado
):
    modelo, resultados_cv, resumen = cargar_adaboost(
        modelo_path,
        busqueda_path,
        resumen_path
    )

    claves_requeridas = {
        "modelo",
        "extractor",
        "scoring",
        "cv",
        "n_splits",
        "n_iter",
        "seed",
        "best_score_cv",
        "best_params",
        "tiempo_busqueda_s",
        "tiempo_refit_s",
        "n_train",
        "dim_embedding",
        "dataset_v2_sha256"
    }

    faltantes = claves_requeridas - set(resumen)

    if faltantes:
        raise ValueError(
            f"Resumen AdaBoost incompleto. Faltan: {sorted(faltantes)}"
        )

    if not isinstance(modelo, AdaBoostClassifier):
        raise TypeError(
            "El artefacto no contiene un AdaBoostClassifier."
        )

    if resumen["extractor"] != "r2plus1d_18_kinetics400":
        raise ValueError(
            "El extractor registrado para AdaBoost no es el esperado."
        )

    if resumen["scoring"] != "average_precision":
        raise ValueError(
            "La métrica registrada para AdaBoost no es Average Precision."
        )

    if resumen["cv"] != "StratifiedGroupKFold":
        raise ValueError(
            "La estrategia de validación de AdaBoost no es la esperada."
        )

    if resumen["n_splits"] != 5:
        raise ValueError(
            "AdaBoost no fue evaluado con cinco folds."
        )

    if resumen["n_iter"] != n_iter_esperado:
        raise ValueError(
            "El número de configuraciones evaluadas no coincide."
        )

    if resumen["seed"] != seed_esperada:
        raise ValueError(
            "La semilla registrada no coincide."
        )

    if resumen["dataset_v2_sha256"] != dataset_sha256_esperado:
        raise ValueError(
            "El SHA-256 del dataset utilizado por AdaBoost no coincide."
        )

    if resumen["n_train"] != n_train_esperado:
        raise ValueError(
            "El número de observaciones de entrenamiento no coincide."
        )

    if resumen["dim_embedding"] != dim_embedding_esperada:
        raise ValueError(
            "La dimensión de los embeddings no coincide."
        )

    if modelo.n_features_in_ != dim_embedding_esperada:
        raise ValueError(
            "El modelo AdaBoost no utiliza la dimensión esperada."
        )

    if len(resultados_cv) != n_iter_esperado:
        raise ValueError(
            "El archivo de resultados CV no contiene el número esperado de configuraciones."
        )

    return {
        "modelo": type(modelo).__name__,
        "configuraciones_cv": len(resultados_cv),
        "dim_embedding": modelo.n_features_in_,
        "best_score_cv": resumen["best_score_cv"]
    }

def inferir_probabilidades_rgb(
    modelo,
    loader,
    device,
    usar_amp=True
):
    modelo.eval()

    y_true = []
    y_prob = []

    amp_habilitado = (
        usar_amp
        and device.type == "cuda"
    )

    with torch.inference_mode():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(
                device,
                non_blocking=True
            )

            with torch.autocast(
                device_type=device.type,
                enabled=amp_habilitado
            ):
                logits = modelo(batch_x)

                probabilidades = torch.softmax(
                    logits,
                    dim=1
                )[:, 1]

            y_true.extend(
                batch_y.cpu().numpy()
            )

            y_prob.extend(
                probabilidades.cpu().numpy()
            )

    return {
        "y_true": np.asarray(
            y_true,
            dtype=np.int64
        ),
        "y_prob": np.asarray(
            y_prob,
            dtype=np.float32
        )
    }

def guardar_predicciones_rgb_validation(
    csv_path,
    resumen_path,
    predicciones,
    resumen
):
    predicciones.to_csv(
        csv_path,
        index=False
    )

    with open(
        resumen_path,
        "w",
        encoding="utf-8"
    ) as archivo:
        json.dump(
            resumen,
            archivo,
            indent=2,
            ensure_ascii=False
        )


def cargar_predicciones_rgb_validation(
    csv_path,
    resumen_path
):
    predicciones = pd.read_csv(
        csv_path
    )

    with open(
        resumen_path,
        "r",
        encoding="utf-8"
    ) as archivo:
        resumen = json.load(
            archivo
        )

    return predicciones, resumen


def validar_predicciones_rgb_validation(
    csv_path,
    resumen_path,
    df_val_esperado,
    dataset_sha256_esperado,
    checkpoint_r2plus1d_h5_path,
    checkpoint_mc3_h5_path
):
    predicciones, resumen = (
        cargar_predicciones_rgb_validation(
            csv_path,
            resumen_path
        )
    )

    columnas_requeridas = [
        "clip_id",
        "grupo_experimental",
        "label",
        "p_r2plus1d_h5",
        "p_mc3_h5"
    ]

    faltantes = set(
        columnas_requeridas
    ) - set(
        predicciones.columns
    )

    if faltantes:
        raise ValueError(
            f"Faltan columnas en predicciones RGB: "
            f"{sorted(faltantes)}"
        )

    if resumen.get(
        "dataset_v2_sha256"
    ) != dataset_sha256_esperado:
        raise ValueError(
            "El SHA-256 del dataset v2 no coincide."
        )

    hashes_requeridos = {
        "checkpoint_r2plus1d_h5_sha256":
            checkpoint_r2plus1d_h5_path,
        "checkpoint_mc3_h5_sha256":
            checkpoint_mc3_h5_path
    }

    for clave_hash, checkpoint_path in hashes_requeridos.items():
        if clave_hash not in resumen:
            raise ValueError(
                f"El resumen de predicciones RGB no contiene {clave_hash}."
            )

        hash_actual = calcular_sha256(checkpoint_path)

        if resumen[clave_hash] != hash_actual:
            raise ValueError(
                f"El hash registrado en {clave_hash} no coincide "
                "con el checkpoint actual."
            )

    if resumen.get(
        "split"
    ) != "validation":
        raise ValueError(
            "El artefacto no corresponde a validation."
        )

    if len(predicciones) != len(
        df_val_esperado
    ):
        raise ValueError(
            "El número de predicciones no coincide "
            "con validation."
        )

    if predicciones[
        "clip_id"
    ].duplicated().any():
        raise ValueError(
            "Existen clip_id duplicados."
        )

    if predicciones[
        columnas_requeridas
    ].isna().any().any():
        raise ValueError(
            "El artefacto contiene valores faltantes."
        )

    for columna in [
        "p_r2plus1d_h5",
        "p_mc3_h5"
    ]:
        valores = predicciones[
            columna
        ].to_numpy(
            dtype=float
        )

        if not np.isfinite(
            valores
        ).all():
            raise ValueError(
                f"{columna} contiene valores no finitos."
            )

        if not (
            (valores >= 0)
            & (valores <= 1)
        ).all():
            raise ValueError(
                f"{columna} contiene probabilidades "
                "fuera de [0, 1]."
            )

    esperado = (
        df_val_esperado[
            [
                "clip_id",
                "grupo_experimental",
                "clase_binaria"
            ]
        ]
        .rename(
            columns={
                "clase_binaria": "label"
            }
        )
        .copy()
    )

    esperado[
        "clip_id"
    ] = esperado[
        "clip_id"
    ].astype(str)

    esperado[
        "grupo_experimental"
    ] = esperado[
        "grupo_experimental"
    ].astype(str)

    predicciones[
        "clip_id"
    ] = predicciones[
        "clip_id"
    ].astype(str)

    predicciones[
        "grupo_experimental"
    ] = predicciones[
        "grupo_experimental"
    ].astype(str)

    comparacion = predicciones[
        [
            "clip_id",
            "grupo_experimental",
            "label"
        ]
    ].merge(
        esperado,
        on="clip_id",
        how="outer",
        suffixes=(
            "_pred",
            "_esperado"
        ),
        indicator=True
    )

    if not comparacion[
        "_merge"
    ].eq(
        "both"
    ).all():
        raise ValueError(
            "Los clip_id no coinciden con validation."
        )

    if not comparacion[
        "grupo_experimental_pred"
    ].eq(
        comparacion[
            "grupo_experimental_esperado"
        ]
    ).all():
        raise ValueError(
            "Los grupos experimentales no coinciden."
        )

    if not comparacion[
        "label_pred"
    ].astype(int).eq(
        comparacion[
            "label_esperado"
        ].astype(int)
    ).all():
        raise ValueError(
            "Las etiquetas no coinciden con validation."
        )

    return {
        "n_observaciones": len(
            predicciones
        ),
        "clips_unicos": predicciones[
            "clip_id"
        ].nunique(),
        "positivos": int(
            predicciones[
                "label"
            ].sum()
        ),
        "p_r2plus1d_finitas": True,
        "p_mc3_finitas": True,
        "alineacion_dataset": True
    }
def validar_blending(
    oof_path,
    modelo_path,
    resumen_path,
    df_validation,
    dataset_sha256,
    seed_esperado=42,
    n_splits_esperado=3
):
    import json
    import joblib

    archivos = [oof_path, modelo_path, resumen_path]

    for ruta in archivos:
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el artefacto requerido: {ruta.name}")

    oof = pd.read_csv(oof_path)

    columnas_requeridas = {
        "clip_id",
        "grupo_experimental",
        "label",
        "fold",
        "p_blending_oof"
    }

    faltantes = columnas_requeridas - set(oof.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en {oof_path.name}: {sorted(faltantes)}"
        )

    if len(oof) != len(df_validation):
        raise ValueError(
            f"Número de observaciones incorrecto: {len(oof)} "
            f"en lugar de {len(df_validation)}."
        )

    if oof["clip_id"].nunique() != len(oof):
        raise ValueError("El artefacto OOF contiene clip_id duplicados.")

    clips_esperados = df_validation["clip_id"].astype(str).to_numpy()
    clips_oof = oof["clip_id"].astype(str).to_numpy()

    if not np.array_equal(clips_oof, clips_esperados):
        raise ValueError(
            "El orden o los clip_id del artefacto OOF no coinciden con validation."
        )

    grupos_esperados = df_validation["grupo_experimental"].astype(str).to_numpy()
    grupos_oof = oof["grupo_experimental"].astype(str).to_numpy()

    if not np.array_equal(grupos_oof, grupos_esperados):
        raise ValueError(
            "Los grupos experimentales del artefacto OOF no coinciden con validation."
        )

    labels_esperados = df_validation["clase_binaria"].to_numpy(dtype=np.int64)
    labels_oof = oof["label"].to_numpy(dtype=np.int64)

    if not np.array_equal(labels_oof, labels_esperados):
        raise ValueError(
            "Las etiquetas del artefacto OOF no coinciden con validation."
        )

    folds = set(oof["fold"].astype(int).unique())

    if folds != set(range(1, n_splits_esperado + 1)):
        raise ValueError(
            f"Folds incompatibles: {sorted(folds)}. "
            f"Se esperaban {list(range(1, n_splits_esperado + 1))}."
        )

    probabilidades = oof["p_blending_oof"].to_numpy(dtype=float)

    if not np.isfinite(probabilidades).all():
        raise ValueError("Las probabilidades OOF contienen valores no finitos.")

    if not ((probabilidades >= 0) & (probabilidades <= 1)).all():
        raise ValueError("Las probabilidades OOF están fuera del intervalo [0, 1].")

    with open(resumen_path, "r", encoding="utf-8") as archivo:
        resumen = json.load(archivo)

    if resumen.get("dataset_v2_sha256") != dataset_sha256:
        raise ValueError("El resumen de Blending corresponde a otro dataset v2.")

    if resumen.get("split") != "validation":
        raise ValueError("El resumen de Blending no corresponde a validation.")

    if int(resumen.get("seed", -1)) != seed_esperado:
        raise ValueError("El seed del artefacto de Blending no coincide.")

    if int(resumen.get("n_splits", -1)) != n_splits_esperado:
        raise ValueError("El número de folds del artefacto de Blending no coincide.")

    if resumen.get("transformacion_entrada") != "logit":
        raise ValueError(
            "El artefacto de Blending no utiliza transformación logit."
        )
    epsilon_logit = resumen.get(
        "epsilon_logit"
    )

    if epsilon_logit is None or not np.isclose(
        float(epsilon_logit),
        1e-6
    ):
        raise ValueError(
            "El epsilon de la transformación logit no coincide."
        )

    if float(resumen.get("C", np.nan)) != 0.1:
        raise ValueError(
            "El valor de C del meta-modelo de Blending no coincide."
        )

    if resumen.get("solver") != "liblinear":
        raise ValueError(
            "El solver del meta-modelo de Blending no coincide."
        )

    modelo = joblib.load(modelo_path)

    if getattr(modelo, "n_features_in_", None) != 2:
        raise ValueError(
            "El meta-modelo de Blending no fue ajustado con exactamente 2 variables."
        )

    return {
        "n_observaciones": len(oof),
        "clips_unicos": int(oof["clip_id"].nunique()),
        "positivos": int(oof["label"].sum()),
        "folds": sorted(folds),
        "probabilidades_finitas": bool(np.isfinite(probabilidades).all()),
        "modelo_n_features": int(modelo.n_features_in_),
        "alineacion_dataset": True
    }

def cargar_predicciones_rgb(
    csv_path,
    resumen_path
):
    predicciones = pd.read_csv(csv_path)

    with open(resumen_path, "r", encoding="utf-8") as archivo:
        resumen = json.load(archivo)

    return predicciones, resumen


def guardar_predicciones_rgb(
    csv_path,
    resumen_path,
    predicciones,
    resumen
):
    predicciones.to_csv(csv_path, index=False)

    with open(resumen_path, "w", encoding="utf-8") as archivo:
        json.dump(
            resumen,
            archivo,
            indent=2,
            ensure_ascii=False
        )

def validar_predicciones_r2_test(
    csv_path,
    resumen_path,
    df_test_esperado,
    dataset_sha256_esperado,
    checkpoint_r2plus1d_h5_path
):
    predicciones, resumen = cargar_predicciones_rgb(
        csv_path,
        resumen_path
    )

    columnas_requeridas = [
        "clip_id",
        "grupo_experimental",
        "label",
        "p_r2plus1d_h5"
    ]

    faltantes = set(columnas_requeridas) - set(predicciones.columns)

    if faltantes:
        raise ValueError(
            f"Faltan columnas en predicciones de test: {sorted(faltantes)}"
        )

    if resumen.get("dataset_v2_sha256") != dataset_sha256_esperado:
        raise ValueError(
            "El SHA-256 del dataset v2 no coincide."
        )

    if resumen.get("split") != "test":
        raise ValueError(
            "El artefacto no corresponde a test."
        )

    if "checkpoint_r2plus1d_h5_sha256" not in resumen:
        raise ValueError(
            "El resumen de test no contiene el hash del checkpoint."
        )

    hash_checkpoint_actual = calcular_sha256(
        checkpoint_r2plus1d_h5_path
    )

    if (
        resumen["checkpoint_r2plus1d_h5_sha256"]
        != hash_checkpoint_actual
    ):
        raise ValueError(
            "El hash registrado no coincide con el checkpoint "
            "R(2+1)D-18/H5 actual."
        )

    if len(predicciones) != len(df_test_esperado):
        raise ValueError(
            "El número de predicciones no coincide con test."
        )

    if predicciones["clip_id"].duplicated().any():
        raise ValueError(
            "Existen clip_id duplicados en las predicciones de test."
        )

    if predicciones[columnas_requeridas].isna().any().any():
        raise ValueError(
            "El artefacto de test contiene valores faltantes."
        )

    probabilidades = predicciones[
        "p_r2plus1d_h5"
    ].to_numpy(dtype=float)

    if not np.isfinite(probabilidades).all():
        raise ValueError(
            "Las probabilidades de test contienen valores no finitos."
        )

    if not (
        (probabilidades >= 0)
        & (probabilidades <= 1)
    ).all():
        raise ValueError(
            "Las probabilidades de test están fuera de [0, 1]."
        )

    esperado = (
        df_test_esperado[
            [
                "clip_id",
                "grupo_experimental",
                "clase_binaria"
            ]
        ]
        .rename(columns={"clase_binaria": "label"})
        .copy()
    )

    esperado["clip_id"] = esperado["clip_id"].astype(str)
    esperado["grupo_experimental"] = (
        esperado["grupo_experimental"].astype(str)
    )

    predicciones["clip_id"] = predicciones["clip_id"].astype(str)
    predicciones["grupo_experimental"] = (
        predicciones["grupo_experimental"].astype(str)
    )

    comparacion = predicciones[
        [
            "clip_id",
            "grupo_experimental",
            "label"
        ]
    ].merge(
        esperado,
        on="clip_id",
        how="outer",
        suffixes=("_pred", "_esperado"),
        indicator=True
    )

    if not comparacion["_merge"].eq("both").all():
        raise ValueError(
            "Los clip_id no coinciden con test."
        )

    if not comparacion[
        "grupo_experimental_pred"
    ].eq(
        comparacion["grupo_experimental_esperado"]
    ).all():
        raise ValueError(
            "Los grupos experimentales no coinciden con test."
        )

    if not comparacion[
        "label_pred"
    ].astype(int).eq(
        comparacion["label_esperado"].astype(int)
    ).all():
        raise ValueError(
            "Las etiquetas no coinciden con test."
        )

    return {
        "n_observaciones": len(predicciones),
        "clips_unicos": predicciones["clip_id"].nunique(),
        "positivos": int(predicciones["label"].sum()),
        "probabilidades_finitas": True,
        "alineacion_dataset": True
    }